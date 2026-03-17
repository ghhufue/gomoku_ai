#pragma once

#include <array>
#include <cstdint>

namespace gomoku {

constexpr int kDirectionRadius = 5;
constexpr int kDirectionLineLength = 2 * kDirectionRadius + 1;
constexpr int kDirectionSideCount = 2 * kDirectionRadius;
/**
 * @brief Total size of the compact lookup-key space.
 *
 * Keys are laid out as [length bucket + left-trim count + base-3 pattern code].
 */
constexpr int kDirectionLookupTableSize = 132854;

/**
 * @brief Absolute stone values stored on the board.
 */
enum class Stone : std::int8_t {
    kEmpty = 0,
    kBlack = 1,
    kWhite = -1,
};

/**
 * @brief Board coordinate in row/column form.
 */
struct Vec2i {
    int row = 0;
    int col = 0;
};

/**
 * @brief Canonical scan directions used by directional pattern extraction.
 */
enum class Direction : std::uint8_t {
    kHorizontal = 0,
    kVertical = 1,
    kMainDiagonal = 2,
    kAntiDiagonal = 3,
};

/**
 * @brief Relative cell state from the current player's perspective.
 */
enum class RelativeCellState : std::uint8_t {
    kEmpty = 0,
    kSelf = 1,
    kOther = 2,
};

/**
 * @brief Raw directional line centered on the candidate move.
 */
struct DirectionLine {
    std::array<RelativeCellState, kDirectionLineLength> cells{};
};

/**
 * @brief Encoded representation used by the direction lookup table.
 */
struct EncodedDirectionState {
    std::array<RelativeCellState, kDirectionSideCount> side_cells{};
    std::uint8_t effective_length = 0;
    std::uint8_t left_trimmed_empty = 0;
    std::uint8_t right_trimmed_empty = 0;
    std::uint32_t pattern_code = 0;
    std::uint32_t lookup_key = 0;
};

/**
 * @brief Powers of three used by base-3 encoding and decoding.
 */
constexpr std::array<std::uint32_t, 11> kPow3ByLength = {
    1u,
    3u,
    9u,
    27u,
    81u,
    243u,
    729u,
    2187u,
    6561u,
    19683u,
    59049u,
};

/**
 * @brief Starting offset of each effective-length bucket.
 *
 * Shorter effective sequences are assigned smaller key ranges.
 */
constexpr std::array<std::uint32_t, 11> kLookupBucketOffsets = {
    0u,
    11u,
    41u,
    122u,
    338u,
    905u,
    2363u,
    6008u,
    14756u,
    34439u,
    93488u,
};

/**
 * @brief Extract a directional line around a board position.
 *
 * The center cell is treated as the current player's move. Out-of-board cells
 * are normalized to `kEmpty`.
 *
 * @param[in] board Pointer to a row-major board buffer.
 * @param[in] board_size Board edge length. Defaults to 15.
 * @param[in] center Center position of the extracted line.
 * @param[in] direction Scan direction.
 * @param[in] self_stone Stone treated as `kSelf`.
 * @return Extracted directional line.
 * @warning `board` must not be null and `center` must be inside the board.
 */
DirectionLine extract_direction_line(
    const std::int8_t* board,
    Vec2i center,
    Direction direction,
    Stone self_stone,
    int board_size = 15
);

/**
 * @brief Drop the center cell and build the 10-cell side-state array.
 *
 * The output order is fixed as `[-5..-1, +1..+5]`.
 *
 * @param[in] line Raw directional line.
 * @return Side-state array used by the encoder.
 * @see extract_direction_line
 */
std::array<RelativeCellState, kDirectionSideCount> line_to_side_cells(const DirectionLine& line);

/**
 * @brief Encode a 10-cell side-state array into a compact lookup key.
 *
 * The encoder trims empty cells on both ends and then base-3 encodes the
 * remaining sequence.
 *
 * @param[in] side_cells Ten relative states ordered as `[-5..-1, +1..+5]`.
 * @return Structured encoding result, including the final lookup key.
 */
EncodedDirectionState encode_direction_side_cells(const std::array<RelativeCellState, kDirectionSideCount>& side_cells);

/**
 * @brief Decode a lookup key back into its structured form.
 *
 * This helper is intended for debugging and table validation.
 *
 * @param[in] lookup_key Encoded lookup key.
 * @return Decoded directional state.
 * @warning `lookup_key` must be smaller than `kDirectionLookupTableSize`.
 * @see encode_direction_side_cells
 */
EncodedDirectionState decode_direction_lookup_key(std::uint32_t lookup_key);

}  // namespace gomoku
