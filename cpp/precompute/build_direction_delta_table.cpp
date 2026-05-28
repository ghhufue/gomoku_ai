#include "../direction_encoding.h"
#include "../StateValueRegistry.h"
#include "../utils/direction_pattern_lookup.h"
#include "DirectionDeltaTable.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gomoku::precompute {

namespace {

struct DirectionDeltaTableFileHeader {
    char magic[8] = {'G', 'M', 'D', 'L', 'T', 'B', '1', '\0'};
    std::uint32_t version = 1;
    std::uint32_t entry_count = kDirectionDeltaTableEntryCount;
    std::uint32_t state_count = static_cast<std::uint32_t>(kTrackedStateValueCount);
};

PackedDirectionDeltaEntry build_entry(std::uint32_t key) {
    const auto side_cells = decode_full_direction_side_cells(key);
    const StateValueDeltaVector self_delta =
        classify_direction_state_delta_with_center(side_cells, kDirectionRadius, RelativeCellState::kSelf);
    const StateValueDeltaVector other_delta =
        classify_direction_state_delta_with_center(side_cells, kDirectionRadius, RelativeCellState::kOther);

    PackedDirectionDeltaEntry entry{};
    for (std::size_t index = 0; index < kTrackedStateValueCount; ++index) {
        entry.self_delta[index] = static_cast<std::int8_t>(self_delta[index].delta);
        entry.other_delta[index] = static_cast<std::int8_t>(other_delta[index].delta);
    }
    return entry;
}

}  // namespace

int build_direction_delta_table_file(
    const std::string& output_path = "d:/code/gomoku_ai/cpp/precompute/direction_delta_table.bin"
) {
    std::ofstream output(output_path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open output file: " + output_path);
    }

    const DirectionDeltaTableFileHeader header{};
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!output.good()) {
        throw std::runtime_error("failed to write table header");
    }

    for (std::uint32_t key = 0; key < kDirectionDeltaTableEntryCount; ++key) {
        const PackedDirectionDeltaEntry entry = build_entry(key);
        output.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
        if (!output.good()) {
            throw std::runtime_error("failed to write table payload");
        }
    }

    return 0;
}

}  // namespace gomoku::precompute

int main() {
    try {
        return gomoku::precompute::build_direction_delta_table_file();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
