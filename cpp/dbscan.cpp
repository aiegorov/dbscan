#include "cpp/dbscan.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <climits>
#include <unordered_map>

namespace dbscan {

Dbscan::Dbscan(float const eps, std::uint32_t const min_samples, std::size_t const num_points_hint)
    : eps_squared_{eps * eps}
    , min_samples_{min_samples}
    , max_durations_window_{1000}
{
    if (num_points_hint > 0) {
        labels_.reserve(num_points_hint);
        neighbors_.reserve(num_points_hint);
        visited_.reserve(num_points_hint);
        to_visit_.reserve(num_points_hint);
    }
}

void Dbscan::update_durations_(std::uint32_t new_duration){

    if (durations_.size() >= max_durations_window_ - 1){
        durations_.pop_front();
    }
    durations_.push_back(new_duration);
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

    std::vector<std::uint32_t> num_neighbors(std::size(new_points), 0);

    const auto n_points{points.size()};

    std::vector<std::array<std::int32_t, 3>> core_points_ids;
    core_points_ids.assign(new_points.size(), {-1, -1, -1});

    const auto now_1 = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (auto i = 0UL; i < std::size(new_points); ++i) {
        auto const& pt = new_points[i];
        auto const bin_x = static_cast<std::int32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::int32_t>(std::floor((pt[1] - min[1]) / eps));

        std::vector<std::uint32_t> local_neighbors;

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
                auto const & neighbor_pt = new_points[neighbor_pt_index];
                if ((square(neighbor_pt[0] - pt[0]) + square(neighbor_pt[1] - pt[1])) < eps_squared_) {
                    local_neighbors.push_back(neighbor_pt_index);
                }
            }
        }
        if (std::size(local_neighbors) > min_samples_) {
            for (auto const n : local_neighbors) {
                for (auto cp_id{0U}; cp_id < core_points_ids.at(n).size(); ++cp_id) {
                    if (core_points_ids.at(n).at(cp_id) == -1) {
                        core_points_ids.at(n).at(cp_id) = i;
                        break;
                    }
                }
            }
        }
    }
    const auto now_2 = std::chrono::system_clock::now();
    const std::uint32_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(now_2 - now_1).count();
    std::cerr << "Block in question took " << duration << " ms" << std::endl;
    update_durations_(duration);
    std::cerr << "Mean duration is " << std::accumulate(durations_.begin(), durations_.end(), 0) / durations_.size() << " ms" << std::endl;

    for (auto i{0UL}; i < new_points.size(); ++i) {
        if (core_points_ids.at(i).at(0) >= 0) {
            labels_.at(i) = static_cast<Label>(i);
        } else {
            labels_.at(i) = noise;
        }
    }

    bool converged{false};
    int num_iterations{0};

    while (!converged) {
        num_iterations++;
        converged = true;
        for (auto i{0UL}; i < new_points.size(); ++i) {
            if (labels_.at(i) == -1) continue;
            for (const auto current_core_idx : core_points_ids.at(i)){
                if (current_core_idx == -1) continue;
                if (labels_.at(i) < labels_.at(current_core_idx)) {
                    labels_.at(current_core_idx) = labels_.at(i);
                    converged = false;
                } else if (labels_.at(i) > labels_.at(current_core_idx)) {
                    labels_.at(i) = labels_.at(current_core_idx);
                    converged = false;
                }
            }
        }
    }
    std::cerr << "converged in " << num_iterations << " iterations" << std::endl;

    std::unordered_map<Label, Label> labels_map;
    labels_map.reserve(labels_.size());

    Label num_labels{0};
    for (const auto l : labels_){
        if (! labels_map.contains(l)) {
            labels_map[l] = num_labels;
            num_labels++;
        }
    }
    labels_map[noise] = noise;

//
//
//    std::vector<int32_t> labels_bin_vector(std::size(labels_), 0);
//    for (const auto& class_label : labels_) {
//        if (class_label != undefined && class_label != noise) {
//            labels_bin_vector.at(static_cast<uint32_t>(class_label)) = 1;
//        }
//    }
//    std::vector<uint32_t> real_class_ids_2_new_class_ids;
//    real_class_ids_2_new_class_ids.reserve(labels_bin_vector.size());
//    std::inclusive_scan(
//        labels_bin_vector.begin(), labels_bin_vector.end(), std::back_inserter(real_class_ids_2_new_class_ids));

    std::vector<Label> labels(std::size(labels_));
    for (auto i{0U}; i < std::size(labels_); ++i) {
        labels.at(new_point_to_point_index_map.at(i)) = labels_map[labels_.at(i)];
    }

    std::cerr << "returning labels" << std::endl;

    return labels;
}

}  // namespace dbscan
