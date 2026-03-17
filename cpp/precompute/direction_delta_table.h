#pragma once

#include "../direction_encoding.h"
#include "../state_value.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gomoku {

constexpr std::uint32_t kDirectionDeltaTableEntryCount = 59049;

struct PackedDirectionDeltaEntry {
    std::array<std::int8_t, kTrackedStateValueCount> self_delta{};
    std::array<std::int8_t, kTrackedStateValueCount> other_delta{};
};

class DirectionDeltaTable {
public:
    std::uint32_t encodeKey(const std::array<RelativeCellState, kDirectionSideCount>& side_cells) const;
    std::array<RelativeCellState, kDirectionSideCount> decodeKey(std::uint32_t lookup_key) const;
    const PackedDirectionDeltaEntry& lookup(const std::array<RelativeCellState, kDirectionSideCount>& side_cells);
    const PackedDirectionDeltaEntry& lookup(std::uint32_t lookup_key);
    std::string defaultPath() const;
    void loadFromFile(const std::string& path);
    bool isLoaded() const;

    static DirectionDeltaTable& instance();

private:
    void ensureLoaded();

    std::vector<PackedDirectionDeltaEntry> table_{};
};

std::uint32_t encode_full_direction_side_cells(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells
);

std::array<RelativeCellState, kDirectionSideCount> decode_full_direction_side_cells(std::uint32_t lookup_key);

const PackedDirectionDeltaEntry& lookup_direction_delta_entry(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells
);

const PackedDirectionDeltaEntry& lookup_direction_delta_entry(std::uint32_t lookup_key);

std::string default_direction_delta_table_path();

void load_direction_delta_table_from_file(const std::string& path);

bool is_direction_delta_table_loaded();

}  // namespace gomoku
