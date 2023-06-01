#include "cpp/dbscan.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>

namespace dbscan {

Dbscan::Dbscan(float const eps,
               std::uint32_t const min_samples,
               const std::vector<float> x_slices,
               std::size_t const num_points_hint)
    : eps_squared_{eps * eps}
    , min_samples_{min_samples}
    , x_slices_{x_slices}
{
    labels_outputs.reserve(x_slices_.size() - 1);
    points_in_slices.reserve(x_slices_.size() - 1);
    idx.reserve(x_slices_.size() - 1);
}

auto Dbscan::fit_predict(std::vector<Dbscan::Point> const& points) -> std::vector<Dbscan::Label>
{

    auto ts1 = std::chrono::high_resolution_clock::now();
    points_in_slices.clear();
//    labels_outputs.clear();
    labels_outputs.resize(x_slices_.size() - 1);
    idx.clear();

    for (uint32_t i = 0; i < x_slices_.size() - 1; ++i) {
//        std::vector<Dbscan::Point> points_;
//        points_.reserve(
        points_in_slices.emplace_back();
        points_in_slices.back().reserve(points.size());

//        std::vector<std::uint32_t> idx_vec;
        idx.emplace_back();
        idx.back().reserve(points.size());
//        idx_vec.reserve(points.size());
//        idx.push_back(idx_vec);
    }

    auto ts2 = std::chrono::high_resolution_clock::now();
    std::cerr << "resizes took " << std::chrono::duration_cast<std::chrono::microseconds>(ts2 - ts1).count() << "mcs" << std::endl;

//    labels_outputs.reserve(x_slices_.size() - 1);

    for (uint32_t i = 0; i < points.size(); ++i) {
        auto& point{points[i]};
        for (size_t j = 0; j < x_slices_.size() - 1; ++j) {
            if (point.at(0) >= x_slices_.at(j) && point.at(0) < x_slices_[j + 1]) {
                points_in_slices.at(j).push_back(point);
                idx.at(j).push_back(i);
                break;
            }
        }
    }

    auto ts3 = std::chrono::high_resolution_clock::now();
    std::cerr << "sorting took " << std::chrono::duration_cast<std::chrono::microseconds>(ts3 - ts2).count() << "mcs" << std::endl;

// this loop will be parallelized
size_t i{0};
#pragma omp parallel num_threads(8)
    for (i = 0; i < x_slices_.size() - 1; ++i) {
        labels_outputs.at(i) = fit_predict_single(points_in_slices.at(i));

//        labels_outputs.push_back(fit_predict_single(points_in_slice));
    }
    auto ts4 = std::chrono::high_resolution_clock::now();
    std::cerr << "execution took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts4 - ts3).count() << "ms" << std::endl;

    // TODO this is only correct assuming all the points are still within our partitions
    std::vector<Dbscan::Label> labels_final(points.size());
    labels_final.assign(points.size(), undefined);

    Label last_max{-1};

    for (size_t i{0}; i < labels_outputs.size(); ++i) {
        for (auto& label : labels_outputs.at(i)) {
            if (i != 0 && label != noise) {
                label += last_max + 1;
            }
        }

        if (labels_outputs.at(i).size() > 0) {
            last_max = *std::max_element(labels_outputs.at(i).begin(), labels_outputs.at(i).end());
        }
    }

    for (size_t i{0}; i < labels_outputs.size(); ++i) {
        for (size_t j{0}; j < labels_outputs.at(i).size(); ++j) {
            labels_final.at(idx.at(i).at(j)) = labels_outputs.at(i).at(j);
        }
    }

    auto const& res{labels_final};
//    std::cerr << "And the result will have size: " << res.size() << std::endl;

    return res;
}

auto Dbscan::fit_predict_single(std::vector<Dbscan::Point> const& points)  //, std::vector<Dbscan::Label>& labels_slice)
    -> std::vector<Dbscan::Label>
{
    std::vector<Dbscan::Label> labels_slice;
    labels_slice.assign(std::size(points), undefined);
//    std::cerr << "Size of input points" << points.size() << std::endl;

    if (std::size(points) <= 1) {
        return labels_slice;
    }

    Label cluster_count{0};
    std::vector<Label> clusters{};

    // reorg point cloud

    auto const eps = std::sqrt(eps_squared_);

    Dbscan::Point min{points[0]};
    Dbscan::Point max{points[0]};
    for (auto const& pt : points) {
        min[0] = std::min(min[0], pt[0]);
        min[1] = std::min(min[1], pt[1]);
        max[0] = std::max(max[0], pt[0]);
        max[1] = std::max(max[1], pt[1]);
    }

    float const range_x = max[0] - min[0];
    float const range_y = max[1] - min[1];
    auto const num_bins_x = static_cast<std::uint32_t>(std::ceil(range_x / eps));
    auto const num_bins_y = static_cast<std::uint32_t>(std::ceil(range_y / eps));

    std::vector<std::uint32_t> counts(num_bins_x * num_bins_y);

    for (auto const& pt : points) {
        auto const bin_x = static_cast<std::uint32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::uint32_t>(std::floor((pt[1] - min[1]) / eps));
        auto const index = bin_y * num_bins_x + bin_x;
        counts[index] += 1;
    }

    std::vector<std::uint32_t> offsets{};
    offsets.reserve(std::size(counts));
    std::exclusive_scan(std::cbegin(counts), std::cend(counts), std::back_inserter(offsets), 0);

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

    std::vector<std::uint32_t> neighbors;
    neighbors.reserve(16364);

    auto square = [](float const v) -> float {
        return v * v;
    };

    auto radius_search = [&](std::uint32_t pt_index) -> std::uint32_t {
        neighbors.clear();
        auto const& pt = new_points[pt_index];
        auto const bin_x = static_cast<std::int32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::int32_t>(std::floor((pt[1] - min[1]) / eps));
        for (auto neighbor_bin_y = bin_y - 1; neighbor_bin_y <= bin_y + 1; ++neighbor_bin_y) {
            for (auto neighbor_bin_x = bin_x - 1; neighbor_bin_x <= bin_x + 1; ++neighbor_bin_x) {
                if (neighbor_bin_x < 0 || neighbor_bin_x >= num_bins_x || neighbor_bin_y < 0 ||
                    neighbor_bin_y >= num_bins_y) {
                    continue;
                }
                auto const neighbor_bin = neighbor_bin_y * num_bins_x + neighbor_bin_x;
                for (auto i{0U}; i < counts[neighbor_bin]; ++i) {
                    auto const neighbor_pt_index = offsets[neighbor_bin] + i;
                    if (neighbor_pt_index == pt_index /*|| visited_[neighbor_pt_index]*/) {
                        continue;
                    }
                    auto const neighbor_pt = new_points[neighbor_pt_index];
                    if ((square(neighbor_pt[0] - pt[0]) + square(neighbor_pt[1] - pt[1])) < eps_squared_) {
                        neighbors.push_back(neighbor_pt_index);
                    }
                }
            }
        }
        return neighbors.size();
    };

    for (auto i{0UL}; i < std::size(new_points); ++i) {
        // skip point if it has already been processed
        if (labels_slice[i] != undefined) {
            continue;
        }

        // find number of neighbors of current point
        if (radius_search(i) < min_samples_) {
            labels_slice[i] = noise;
            continue;
        }

        // std::cout << "cont" << std::endl;

        // This point has at least min_samples_ in its eps neighborhood, so it's considered a core point. Time to
        // start a new cluster.

        auto const current_cluster_id{cluster_count++};
        labels_slice[i] = current_cluster_id;

        to_visit_.clear();
        visited_.assign(std::size(new_points), false);

        for (auto const& n : neighbors) {
            if (!visited_[n]) {
                to_visit_.push_back(n);
            }
            visited_[n] = true;
        }

        for (auto j{0UL}; j < std::size(to_visit_); ++j) {
            auto const neighbor{to_visit_[j]};

            if (labels_slice[neighbor] == noise) {
                // This was considered as a seed before, but didn't have enough points in its eps neighborhood.
                // Since it's in the current seed's neighborhood, we label it as belonging to this label, but it
                // won't be used as a seed again.
                labels_slice[neighbor] = current_cluster_id;
                continue;
            }

            if (labels_slice[neighbor] != undefined) {
                // Point belongs already to a cluster: skip it.
                continue;
            }

            // assign the current cluster's label to the neighbor
            labels_slice[neighbor] = current_cluster_id;

            // and query its neighborhood to see if it also to be considered as a core point
            if (radius_search(neighbor) < min_samples_) {
                continue;
            }
            for (auto const& n : neighbors) {
                if (!visited_[n]) {
                    to_visit_.push_back(n);
                }
                visited_[n] = true;
            }
        }
    }

    std::vector<Label> labels(std::size(labels_slice));
    for (auto i{0U}; i < std::size(labels_slice); ++i) {
        labels[new_point_to_point_index_map[i]] = labels_slice[i];
    }

//    std::cerr << "Output will have the size " << labels.size() << std::endl;

    return labels;
}

}  // namespace dbscan
