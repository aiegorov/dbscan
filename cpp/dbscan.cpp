#include "cpp/dbscan.hpp"

#include <nanoflann.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>

namespace dbscan {

namespace {

class PointsVectorAdaptor
{
public:
    explicit PointsVectorAdaptor(std::vector<Dbscan::Point> const& points)
        : points_{points}
    {
    }

    std::size_t kdtree_get_point_count() const
    {
        return std::size(points_);
    }

    float kdtree_get_pt(std::size_t idx, int dim) const
    {
        return points_[idx][dim];
    }

    template <typename Bbox>
    bool kdtree_get_bbox(Bbox&) const
    {
        return false;
    }

private:
    std::vector<Dbscan::Point> const& points_;
};

}  // namespace

Dbscan::Dbscan(float const eps,
               std::uint32_t const min_samples,
               const std::vector<float> x_slices,
               std::size_t const num_points_hint)
    : eps_squared_{eps * eps}
    , min_samples_{min_samples}
    , x_slices_{x_slices}
{
    if (num_points_hint > 0) {
//        labels_.reserve(num_points_hint);
        neighbors_.reserve(num_points_hint);
        visited_.reserve(num_points_hint);
        to_visit_.reserve(num_points_hint);
    }

    labels_outputs.resize(x_slices_.size() - 1);
    points_in_slices.resize(x_slices_.size() - 1);
    idx.resize(x_slices_.size() - 1);
}

auto Dbscan::fit_predict(std::vector<Dbscan::Point> const& points) -> std::vector<Dbscan::Label>
{

    auto ts1 = std::chrono::high_resolution_clock::now();
    points_in_slices.clear();
//    labels_outputs.clear();
//    labels_outputs.resize(x_slices_.size() - 1);
    idx.clear();

    for (uint32_t i = 0; i < x_slices_.size() - 1; ++i) {
//        std::vector<Dbscan::Point> points_;
//        points_.reserve(
        points_in_slices.emplace_back();
        points_in_slices.back().reserve(points.size());

//        std::vector<std::uint32_t> idx_vec;
        idx.emplace_back();
        idx.back().reserve(points.size());

        labels_outputs.at(i).reserve(points.size());
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
#pragma omp parallel for
    for (size_t i = 0; i < x_slices_.size() - 1; ++i) {
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

auto Dbscan::fit_predict_single(std::vector<Dbscan::Point> const& points) -> std::vector<Dbscan::Label>
{
    PointsVectorAdaptor adapter{points};

    std::vector<Label> labels_;

    constexpr auto num_dims{2};
    constexpr auto leaf_size{32};
    nanoflann::
        KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<float, PointsVectorAdaptor>, PointsVectorAdaptor, num_dims>
            points_kd_tree{num_dims, adapter, nanoflann::KDTreeSingleIndexAdaptorParams{leaf_size}};
    points_kd_tree.buildIndex();

    nanoflann::SearchParams params{};
    params.sorted = false;

    labels_.assign(std::size(points), undefined);

    Label cluster_count{0};
    std::vector<Label> clusters{};

    for (auto i{0UL}; i < std::size(points); ++i) {
        // skip point if it has already been processed
        if (labels_[i] != undefined) {
            continue;
        }

        // find number of neighbors of current point
        if (points_kd_tree.radiusSearch(points[i].data(), eps_squared_, neighbors_, params) < min_samples_) {
            labels_[i] = noise;
            continue;
        }

        // This point has at least min_samples_ in its eps neighborhood, so it's considered a core point. Time to
        // start a new cluster.

        auto const current_cluster_id{cluster_count++};
        labels_[i] = current_cluster_id;

        to_visit_.clear();
        visited_.assign(std::size(points), false);

        for (auto const& n : neighbors_) {
            if (!visited_[n.first]) {
                to_visit_.push_back(n.first);
            }
            visited_[n.first] = true;
        }

        for (auto j{0UL}; j < std::size(to_visit_); ++j) {
            auto const neighbor{to_visit_[j]};

            if (labels_[neighbor] == noise) {
                // This was considered as a seed before, but didn't have enough points in its eps neighborhood.
                // Since it's in the current seed's neighborhood, we label it as belonging to this label, but it
                // won't be used as a seed again.
                labels_[neighbor] = current_cluster_id;
                continue;
            }

            if (labels_[neighbor] != undefined) {
                // Point belongs already to a cluster: skip it.
                continue;
            }

            // assign the current cluster's label to the neighbor
            labels_[neighbor] = current_cluster_id;

            // and query its neighborhood to see if it also to be considered as a core point
            if (points_kd_tree.radiusSearch(points[neighbor].data(), eps_squared_, neighbors_, params) < min_samples_) {
                continue;
            }
            for (auto const& n : neighbors_) {
                if (!visited_[n.first]) {
                    to_visit_.push_back(n.first);
                }
                visited_[n.first] = true;
            }
        }
    }

    return labels_;
}

}  // namespace dbscan
