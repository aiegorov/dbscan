#include "cpp/dbscan.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <climits>

namespace dbscan {

Dbscan::Dbscan(float const eps, std::uint32_t const min_samples, std::size_t const num_points_hint)
    : eps_squared_{eps * eps}
    , min_samples_{min_samples}
{
    if (num_points_hint > 0) {
        labels_.reserve(num_points_hint);
        neighbors_.reserve(num_points_hint);
        visited_.reserve(num_points_hint);
        to_visit_.reserve(num_points_hint);
    }
}

auto Dbscan::fit_predict(std::vector<Dbscan::Point> const& points) -> std::vector<Dbscan::Label>
{
    std::cerr << "Got points" << std::endl;
    labels_.assign(std::size(points), undefined);
    visited_.assign(std::size(points), false);

    if (std::size(points) <= 1) {
        return labels_;
    }

    Label cluster_count{0};
    std::vector<Label> clusters{};

    // reorg point cloud

    auto const eps = std::sqrt(eps_squared_);

    // calculate min_max of the current point cloud
    Dbscan::Point min{points[0]};
    Dbscan::Point max{points[0]};
    for (auto const& pt : points) {
        min[0] = std::min(min[0], pt[0]);
        min[1] = std::min(min[1], pt[1]);
        max[0] = std::max(max[0], pt[0]);
        max[1] = std::max(max[1], pt[1]);
    }

    // derive num_bins out of it
    float const range_x = max[0] - min[0];
    float const range_y = max[1] - min[1];
    auto const num_bins_x = static_cast<std::uint32_t>(std::ceil(range_x / eps));
    auto const num_bins_y = static_cast<std::uint32_t>(std::ceil(range_y / eps));

    // count number of points in every bin
    std::vector<std::uint32_t> counts(num_bins_x * num_bins_y);

    // FIRST PASS OVER THE POINTS
    for (auto const& pt : points) {
        auto const bin_x = static_cast<std::uint32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::uint32_t>(std::floor((pt[1] - min[1]) / eps));
        auto const index = bin_y * num_bins_x + bin_x;
        counts[index] += 1;
    }

    // calculate the offsets for each cell (bin)
    std::vector<std::uint32_t> offsets{};
    offsets.reserve(std::size(counts));
    std::exclusive_scan(std::cbegin(counts), std::cend(counts), std::back_inserter(offsets), 0);

    // re-sorting the points (calculating index mapping) based on the bin indices
    auto scratch = offsets;
    std::vector<Point> new_points(std::size(points));
    std::vector<std::uint32_t> new_point_to_point_index_map(std::size(points));
    std::uint32_t i{0};
    for (auto const& pt : points) {
        auto const bin_x = static_cast<std::uint32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::uint32_t>(std::floor((pt[1] - min[1]) / eps));
        auto const index = bin_y * num_bins_x + bin_x;
        auto new_pt_index = scratch[index];
        scratch[index] += 1;
        new_points[new_pt_index] = pt;
        new_point_to_point_index_map[new_pt_index] = i++;
    }

    //  figuring out which points are the neighbors

    static std::array<std::vector<std::uint32_t>, 32> neighbors;
    for (auto i = 0; i < 32; ++i) neighbors[i].reserve(16364);

    auto square = [](float const v) -> float {
        return v * v;
    };

    //     auto radius_search = [&](std::uint32_t pt_index) -> std::uint32_t {
    //         neighbors.clear();
    //         auto const& pt = new_points[pt_index];
    //         auto const bin_x = static_cast<std::int32_t>(std::floor((pt[0] - min[0]) / eps));
    //         auto const bin_y = static_cast<std::int32_t>(std::floor((pt[1] - min[1]) / eps));
    //         for (auto neighbor_bin_y = bin_y - 1; neighbor_bin_y <= bin_y + 1; ++neighbor_bin_y) {
    //             for (auto neighbor_bin_x = bin_x - 1; neighbor_bin_x <= bin_x + 1; ++neighbor_bin_x) {
    //                 if (neighbor_bin_x < 0 || neighbor_bin_x >= num_bins_x || neighbor_bin_y < 0 ||
    //                     neighbor_bin_y >= num_bins_y) {
    //                     continue;
    //                 }
    //                 auto const neighbor_bin = neighbor_bin_y * num_bins_x + neighbor_bin_x;
    //                 for (auto i{0U}; i < counts[neighbor_bin]; ++i) {
    //                     auto const neighbor_pt_index = offsets[neighbor_bin] + i;
    //                     if (neighbor_pt_index == pt_index /*|| visited_[neighbor_pt_index]*/) {
    //                         continue;
    //                     }
    //                     auto const neighbor_pt = new_points[neighbor_pt_index];
    //                     if ((square(neighbor_pt[0] - pt[0]) + square(neighbor_pt[1] - pt[1])) < eps_squared_) {
    //                         neighbors.push_back(neighbor_pt_index);
    //                     }
    //                 }
    //             }
    //         }
    //         return neighbors.size();
    //     };

    std::vector<std::uint32_t> num_neighbors(std::size(new_points), 0);

    //    #pragma omp parallel for shared(labels_)
    for (auto i = 0UL; i < std::size(new_points); ++i) {
        //    for (uint64_t i = 0; i < std::size(new_points); ++i) {
        auto const& pt = new_points[i];
        auto const bin_x = static_cast<std::int32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::int32_t>(std::floor((pt[1] - min[1]) / eps));
        // constexpr std::array<int, 5> dx = {0, 0, -1, -1, -1};
        // constexpr std::array<int, 5> dy = {0, -1, -1, 0, 1};
        // for (auto ni = 0; ni < 4; ++ni) {
        //     auto const nx = bin_x + dx[ni];
        //     auto const ny = bin_y + dy[ni];
        //     if (nx < 0 || ny < 0 || ny >= num_bins_y) {
        //         continue;
        //     }

        auto& local_neighbors = neighbors[i % 32];
        local_neighbors.clear();
        constexpr std::array<int, 9> dx = {-1, +0, +1, -1, +0, +1, -1, +0, +1};
        constexpr std::array<int, 9> dy = {-1, -1, -1, +0, +0, +0, +1, +1, +1};
        for (auto ni = 0; ni < 9; ++ni) {
            auto const nx = bin_x + dx[ni];
            auto const ny = bin_y + dy[ni];
            if (nx < 0 || ny < 0 || nx >= num_bins_x || ny >= num_bins_y) {
                continue;
            }
            auto const neighbor_bin = ny * num_bins_x + nx;
            for (auto j{0U}; j < counts[neighbor_bin]; ++j) {
                auto const neighbor_pt_index = offsets[neighbor_bin] + j;
                if (neighbor_pt_index == i) {
                    continue;
                }
                auto const neighbor_pt = new_points[neighbor_pt_index];
                if ((square(neighbor_pt[0] - pt[0]) + square(neighbor_pt[1] - pt[1])) < eps_squared_) {
                    // num_neighbors[neighbor_pt_index] += 1;
                    // num_neighbors[i] += 1;
                    local_neighbors.push_back(neighbor_pt_index);
                }
            }
        }
        if (std::size(local_neighbors) > min_samples_) {

            const auto label_to_set = static_cast<Label>(i);  // % INT_MAX);
            uint32_t prints_count{0};

            Label current_min{1000000};
            for (auto const n : local_neighbors) {
                if (labels_[n] != undefined || labels_[n] != noise){
                       current_min = std::min(labels_[n], current_min);
                }
            }
            current_min = std::min(labels_[i], current_min);
            labels_[i] = current_min;

            for (auto const n : local_neighbors) {
                labels_[n] = current_min;
//
//                if (labels_[n] == undefined || labels_[n] == noise) {
//                    std::cout << "Used to be " << labels_[n] << ", will be replaced with " << label_to_set << std::endl;
//                    labels_[n] = current_min;  // i % INT_MAX;
//                } else {
//
//
//                    const auto to_replace = std::min(labels_[n], label_to_set);
//
//                    if (to_replace != labels_[n] && prints_count++ < 10)
//                    std::cout << "Already existing label " << labels_[n] << " will be replaced with " << to_replace << std::endl;
//
//                    labels_[n] =  current_min; //std::min(labels_[n], label_to_set);
//                }
            }
        }
    }




    std::vector<int32_t> labels_bin_vector(std::size(labels_), 0);
    for (const auto & class_label : labels_) {
        if (class_label != undefined && class_label != noise) {
            labels_bin_vector.at(static_cast<uint32_t>(class_label)) = 1;
        }
//        std::cout << "class_label = " << class_label << " size = " << labels_bin_vector.size() << std::endl;
//        assert(class_label < labels_bin_vector.size());
//        assert(class_label >= 0);
//        assert(false);

    }
    std::vector<uint32_t> real_class_ids_2_new_class_ids;
    real_class_ids_2_new_class_ids.reserve(labels_bin_vector.size());
    std::inclusive_scan(
        labels_bin_vector.begin(), labels_bin_vector.end(), std::back_inserter(real_class_ids_2_new_class_ids));

    std::cout << "labels_bin_vector  " << std::endl;
    for (auto const i : labels_bin_vector){
        std::cout << i << std::endl;
    }

    std::cout << "real_class_ids_2_new_class_ids: " << std::endl;
    for (auto const i : real_class_ids_2_new_class_ids){
        std::cout << i << std::endl;
    }
//    std::cout <<
//    std::cerr << "scan completed, size: " << real_class_ids_2_new_class_ids.size() << std::endl;
//    std::cerr << "humber of clusters: " << real_class_ids_2_new_class_ids.back();
    std::cerr << std::endl;

    std::vector<Label> labels(std::size(labels_));
    for (auto i{0U}; i < std::size(labels_); ++i) {
        if (labels_[i] == undefined || labels[i] == noise) {
            labels[new_point_to_point_index_map[i]] = noise;
        } else {
//            std::cout << "Label: " << labels_[i] << std::endl;
//            std::cout << "value to fill: " << real_class_ids_2_new_class_ids[labels_[i]] - 1 << std::endl;
            labels[new_point_to_point_index_map[i]] = real_class_ids_2_new_class_ids[labels_[i]] - 1;
        }
    }

    std::cerr << "returning labels" << std::endl;

    // for debugging purposes
//    std::vector<Label> labels(std::size(labels_));
//    labels.assign(points.size(), 0);

    return labels;
}

}  // namespace dbscan
