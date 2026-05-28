#include "DirectionDeltaTable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gomoku {

namespace {

constexpr char kMagic[] = "GMDLTB1";
constexpr std::uint32_t kVersion = 1;

struct DirectionDeltaTableFileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t entry_count;
    std::uint32_t state_count;
};

DirectionDeltaTableFileHeader make_header() {
    DirectionDeltaTableFileHeader header{};
    for (std::size_t index = 0; index < sizeof(kMagic) - 1; ++index) {
        header.magic[index] = kMagic[index];
    }
    header.version = kVersion;
    header.entry_count = kDirectionDeltaTableEntryCount;
    header.state_count = static_cast<std::uint32_t>(kTrackedStateValueCount);
    return header;
}

bool header_matches(const DirectionDeltaTableFileHeader& header) {
    const DirectionDeltaTableFileHeader expected = make_header();
    for (std::size_t index = 0; index < sizeof(expected.magic); ++index) {
        if (header.magic[index] != expected.magic[index]) {
            return false;
        }
    }
    return (
        header.version == expected.version &&
        header.entry_count == expected.entry_count &&
        header.state_count == expected.state_count
    );
}

}  // namespace

DirectionDeltaTable& DirectionDeltaTable::instance() {
    static DirectionDeltaTable table{};
    return table;
}

std::uint32_t DirectionDeltaTable::encodeKey(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells
) const {
    std::uint32_t key = 0;
    std::uint32_t factor = 1;
    for (RelativeCellState state : side_cells) {
        key += static_cast<std::uint32_t>(state) * factor;
        factor *= 3;
    }
    return key;
}

std::array<RelativeCellState, kDirectionSideCount> DirectionDeltaTable::decodeKey(std::uint32_t lookup_key) const {
    if (lookup_key >= kDirectionDeltaTableEntryCount) {
        throw std::out_of_range("direction delta table key out of range");
    }

    std::array<RelativeCellState, kDirectionSideCount> side_cells{};
    std::uint32_t remaining = lookup_key;
    for (std::size_t index = 0; index < side_cells.size(); ++index) {
        side_cells[index] = static_cast<RelativeCellState>(remaining % 3);
        remaining /= 3;
    }
    return side_cells;
}

const PackedDirectionDeltaEntry& DirectionDeltaTable::lookup(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells
) {
    return lookup(encodeKey(side_cells));
}

const PackedDirectionDeltaEntry& DirectionDeltaTable::lookup(std::uint32_t lookup_key) {
    ensureLoaded();
    if (lookup_key >= table_.size()) {
        throw std::out_of_range("direction delta table key out of range");
    }
    return table_[lookup_key];
}

std::string DirectionDeltaTable::defaultPath() const {
    namespace fs = std::filesystem;
    const fs::path cwd_path = fs::current_path() / "cpp" / "precompute" / "direction_delta_table.bin";
    if (fs::exists(cwd_path)) {
        return cwd_path.string();
    }
    return "d:/code/gomoku_ai/cpp/precompute/direction_delta_table.bin";
}

void DirectionDeltaTable::loadFromFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open direction delta table: " + path);
    }

    DirectionDeltaTableFileHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input.good() || !header_matches(header)) {
        throw std::runtime_error("invalid direction delta table header: " + path);
    }

    std::vector<PackedDirectionDeltaEntry> table(header.entry_count);
    input.read(
        reinterpret_cast<char*>(table.data()),
        static_cast<std::streamsize>(table.size() * sizeof(PackedDirectionDeltaEntry))
    );
    if (input.fail()) {
        throw std::runtime_error("failed to read direction delta table payload: " + path);
    }

    table_ = std::move(table);
}

bool DirectionDeltaTable::isLoaded() const {
    return !table_.empty();
}

void DirectionDeltaTable::ensureLoaded() {
    if (!table_.empty()) {
        return;
    }
    loadFromFile(defaultPath());
}

std::uint32_t encode_full_direction_side_cells(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells
) {
    return DirectionDeltaTable::instance().encodeKey(side_cells);
}

std::array<RelativeCellState, kDirectionSideCount> decode_full_direction_side_cells(std::uint32_t lookup_key) {
    return DirectionDeltaTable::instance().decodeKey(lookup_key);
}

const PackedDirectionDeltaEntry& lookup_direction_delta_entry(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells
) {
    return DirectionDeltaTable::instance().lookup(side_cells);
}

const PackedDirectionDeltaEntry& lookup_direction_delta_entry(std::uint32_t lookup_key) {
    return DirectionDeltaTable::instance().lookup(lookup_key);
}

std::string default_direction_delta_table_path() {
    return DirectionDeltaTable::instance().defaultPath();
}

void load_direction_delta_table_from_file(const std::string& path) {
    DirectionDeltaTable::instance().loadFromFile(path);
}

bool is_direction_delta_table_loaded() {
    return DirectionDeltaTable::instance().isLoaded();
}

}  // namespace gomoku
