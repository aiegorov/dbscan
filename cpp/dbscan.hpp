#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace dbscan {

class Dbscan
{
public:
    using Point = std::array<float, 2>;
    using Label = std::int32_t;

    static constexpr Label undefined{-2};
    static constexpr Label noise{-1};

    Dbscan(float eps,
           std::uint32_t min_samples,
           const std::vector<float> x_slices,
           std::size_t const num_points_hint = 0);

    [[nodiscard]] std::vector<Label> fit_predict(std::vector<Point> const& points);

    [[nodiscard]] std::vector<Label> fit_predict_single(
        std::vector<Point> const& points);  //, std::vector<Label>& labels_slice);

private:
    float eps_squared_;
    std::uint32_t min_samples_;
    std::vector<float> x_slices_;

    // scratch memory we only have as members to avoid allocations for each call to `fit_predict()`
    //    std::vector<std::vector<Label>> labels_slices;
    //    std::vector<Label> labels_;
    std::vector<std::vector<Label>> labels_outputs;
    std::vector<std::vector<Point>> points_in_slices;
    std::vector<std::vector<std::uint32_t>> idx;
    std::vector<std::pair<std::uint32_t, float>> neighbors_;
    std::vector<bool> visited_;
    std::vector<std::uint32_t> to_visit_;
};

}  // namespace dbscan
