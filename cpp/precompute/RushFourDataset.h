#pragma once

#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>

namespace gomoku::precompute {

pybind11::list generate_rush_four_opening_group(std::uint32_t group_count = 1, std::uint32_t seed = 0);
void write_rush_four_opening_group(
    const std::string& output_path = "outputs/build/precompute/rush_four_group.json",
    std::uint32_t group_count = 1,
    std::uint32_t seed = 0
);

}  // namespace gomoku::precompute
