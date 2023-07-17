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
//#include <omp.h>

namespace dbscan {

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

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
    std::vector<unsigned long long> aggregate_pre_process_durations(points.size());
    std::vector<unsigned long long> aggregate_neighbor_search_durations(points.size());
    std::vector<unsigned long long> aggregate_core_point_durations(points.size());
    std::vector<unsigned long long> aggregate_overall_loop_durations(points.size());

    const auto start_fn_ts = std::chrono::system_clock::now();

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

    std::cout << "num_binx, num bin y = " << num_bins_x << ", " << num_bins_y << std::endl;

    // count number of points in every bin
    std::vector<std::uint32_t> counts(num_bins_x * num_bins_y);

    // FIRST PASS OVER THE POINTS
    for (auto const& pt : points) {
        auto const bin_x = static_cast<std::uint32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::uint32_t>(std::floor((pt[1] - min[1]) / eps));
        auto const index = bin_y * num_bins_x + bin_x;
//        std::cout << "Point " << pt[0] << ", " << pt[1] << " got index " << index << std::endl;
        counts[index] += 1;
    }

//    std::cout << "Counts: ";
    std::cerr << "Max counts: " << *std::max_element(counts.begin(), counts.end()) << std::endl;

//    for (const auto & count : counts){
//        std::cout << count << " ";
//    }
//    std::cout << std::endl;

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

    const auto after_sort_ts = std::chrono::system_clock::now();
    const auto resort_duration {std::chrono::duration_cast<std::chrono::microseconds>(after_sort_ts - start_fn_ts).count()};
    std::cerr << "Re-sorting of points took " << resort_duration << " microsecs" << std::endl;


    //  figuring out which points are the neighbors
//
//    static std::array<std::vector<std::uint32_t>, 32> neighbors;
//    for (auto i = 0; i < 32; ++i) neighbors[i].reserve(16364);

    auto square = [](float const v) -> float {
        return v * v;
    };

    std::vector<std::uint32_t> num_neighbors(std::size(new_points), 0);

    const auto n_points{points.size()};

    std::vector<std::array<std::int32_t, 3>> core_points_ids;
    core_points_ids.assign(new_points.size(), {-1, -1, -1});

    constexpr uint32_t n_threads{16};
    
    const auto before_paral_loop_ts = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (auto i = 0UL; i < std::size(new_points); ++i) {
        const auto ts_before_pre_process = std::chrono::system_clock::now();
        auto const& pt = new_points[i];
        auto const bin_x = static_cast<std::int32_t>(std::floor((pt[0] - min[0]) / eps));
        auto const bin_y = static_cast<std::int32_t>(std::floor((pt[1] - min[1]) / eps));

        std::vector<std::uint32_t> local_neighbors;

        constexpr std::array<int, 9> dx = {-1, +0, +1, -1, +0, +1, -1, +0, +1};
        constexpr std::array<int, 9> dy = {-1, -1, -1, +0, +0, +0, +1, +1, +1};
        const auto ts_after_pre_process = std::chrono::system_clock::now();
        const auto pre_process_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(ts_after_pre_process - ts_before_pre_process).count();
        aggregate_pre_process_durations.at(i) = static_cast<unsigned long long>(pre_process_duration);

        const auto before_neighbor_ts = std::chrono::system_clock::now();
        for (auto ni = 0; ni < 9; ++ni) {
            auto const nx = bin_x + dx[ni];
            auto const ny = bin_y + dy[ni];
            if (nx < 0 || ny < 0 || nx >= num_bins_x || ny >= num_bins_y) {
                continue;
            }
            auto const neighbor_bin = ny * num_bins_x + nx;
            constexpr uint32_t neighbors_cap{100};
            uint32_t neighbors_counter{0};

            for (auto j{0U}; j < counts[neighbor_bin]; ++j) {
                if (neighbors_counter < neighbors_cap) {
                    auto const neighbor_pt_index = offsets[neighbor_bin] + j;
                    if (neighbor_pt_index == i) {
                        continue;
                    }
                    auto const & neighbor_pt = new_points[neighbor_pt_index];
                    if ((square(neighbor_pt[0] - pt[0]) + square(neighbor_pt[1] - pt[1])) < eps_squared_) {
                        local_neighbors.push_back(neighbor_pt_index);
                    }
                }
                neighbors_counter++;
            }
        }
        const auto after_neighbor_ts = std::chrono::system_clock::now();
        const auto neighbor_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(after_neighbor_ts - before_neighbor_ts).count();
        aggregate_neighbor_search_durations.at(i) = static_cast<unsigned long long>(neighbor_duration);

        const auto ts_before_cp_part = std::chrono::system_clock::now();
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
        const auto ts_after_cp_part = std::chrono::system_clock::now();
        const auto cp_part_duration  = std::chrono::duration_cast<std::chrono::nanoseconds>(ts_after_cp_part - ts_before_cp_part).count();
        aggregate_core_point_durations.at(i) = static_cast<unsigned long long>(cp_part_duration);

        const auto overall_loop_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(ts_after_cp_part - ts_before_pre_process).count();
        aggregate_overall_loop_durations.at(i) = static_cast<unsigned long long>(overall_loop_duration);
    }


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
    labels_map[noise] = noise;
    for (const auto l : labels_){
        if (labels_map.find(l) == labels_map.end()) {
            labels_map[l] = num_labels;
            num_labels++;
        }
    }

    std::vector<Label> labels(std::size(labels_));
    for (auto i{0U}; i < std::size(labels_); ++i) {
        labels.at(new_point_to_point_index_map.at(i)) = labels_map[labels_.at(i)];
    }

    return labels;
}

}  // namespace dbscan
