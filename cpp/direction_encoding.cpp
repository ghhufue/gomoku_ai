#include "direction_encoding.h"

#include <stdexcept>

namespace gomoku {

namespace {

struct DirectionDelta {
    int dr = 0;
    int dc = 0;
};

DirectionDelta resolve_direction_delta(Direction direction) {
    switch (direction) {
        case Direction::kHorizontal:
            return DirectionDelta{0, 1};
        case Direction::kVertical:
            return DirectionDelta{1, 0};
        case Direction::kMainDiagonal:
            return DirectionDelta{1, 1};
        case Direction::kAntiDiagonal:
            return DirectionDelta{1, -1};
        default:
            throw std::invalid_argument("unsupported direction");
    }
}

std::int8_t stone_to_board_value(Stone stone) {
    return static_cast<std::int8_t>(stone);
}

/**
 * @brief Convert an absolute board cell into a relative state.
 *
 * `kOther` only represents the opponent stone. Out-of-board cells are handled
 * separately by the caller and never reach this function.
 */
RelativeCellState encode_board_cell(std::int8_t board_value, Stone self_stone) {
    if (board_value == static_cast<std::int8_t>(Stone::kEmpty)) {
        return RelativeCellState::kEmpty;
    }
    return board_value == stone_to_board_value(self_stone) ? RelativeCellState::kSelf : RelativeCellState::kOther;
}

}  // namespace

DirectionLine extract_direction_line(
    const std::int8_t* board,
    Vec2i center,
    Direction direction,
    Stone self_stone,
    int board_size
) {
    if (board == nullptr) {
        throw std::invalid_argument("board pointer must not be null");
    }
    if (board_size <= 0) {
        throw std::invalid_argument("board_size must be positive");
    }
    if (center.row < 0 || center.row >= board_size || center.col < 0 || center.col >= board_size) {
        throw std::out_of_range("center position is out of board range");
    }
    if (self_stone == Stone::kEmpty) {
        throw std::invalid_argument("self stone must be black or white");
    }

    const DirectionDelta delta = resolve_direction_delta(direction);
    DirectionLine line{};

    // The center cell is always treated as the newly placed self stone.
    line.cells[static_cast<std::size_t>(kDirectionRadius)] = RelativeCellState::kSelf;

    for (int step = 1; step <= kDirectionRadius; ++step) {
        const int left_row = center.row - step * delta.dr;
        const int left_col = center.col - step * delta.dc;
        if (left_row >= 0 && left_row < board_size && left_col >= 0 && left_col < board_size) {
            line.cells[static_cast<std::size_t>(kDirectionRadius - step)] =
                encode_board_cell(board[left_row * board_size + left_col], self_stone);
        } else {
            line.cells[static_cast<std::size_t>(kDirectionRadius - step)] = RelativeCellState::kEmpty;
        }

        const int right_row = center.row + step * delta.dr;
        const int right_col = center.col + step * delta.dc;
        if (right_row >= 0 && right_row < board_size && right_col >= 0 && right_col < board_size) {
            line.cells[static_cast<std::size_t>(kDirectionRadius + step)] =
                encode_board_cell(board[right_row * board_size + right_col], self_stone);
        } else {
            line.cells[static_cast<std::size_t>(kDirectionRadius + step)] = RelativeCellState::kEmpty;
        }
    }

    return line;
}

std::array<RelativeCellState, kDirectionSideCount> line_to_side_cells(const DirectionLine& line) {
    std::array<RelativeCellState, kDirectionSideCount> side_cells{};
    // The first five entries map to the left side; the last five map to the right side.
    for (int index = 0; index < kDirectionRadius; ++index) {
        side_cells[static_cast<std::size_t>(index)] = line.cells[static_cast<std::size_t>(index)];
    }
    for (int index = 0; index < kDirectionRadius; ++index) {
        side_cells[static_cast<std::size_t>(kDirectionRadius + index)] =
            line.cells[static_cast<std::size_t>(kDirectionRadius + 1 + index)];
    }
    return side_cells;
}

EncodedDirectionState encode_direction_side_cells(const std::array<RelativeCellState, kDirectionSideCount>& side_cells) {
    EncodedDirectionState encoded{};
    encoded.side_cells = side_cells;

    int trimmed_start = 0;
    int trimmed_end = kDirectionSideCount;

    // Trim edge empties so shorter and denser patterns receive smaller keys.
    while (trimmed_start < trimmed_end && side_cells[static_cast<std::size_t>(trimmed_start)] == RelativeCellState::kEmpty) {
        ++trimmed_start;
    }
    while (trimmed_end > trimmed_start && side_cells[static_cast<std::size_t>(trimmed_end - 1)] == RelativeCellState::kEmpty) {
        --trimmed_end;
    }

    encoded.left_trimmed_empty = static_cast<std::uint8_t>(trimmed_start);
    encoded.right_trimmed_empty = static_cast<std::uint8_t>(kDirectionSideCount - trimmed_end);
    encoded.effective_length = static_cast<std::uint8_t>(trimmed_end - trimmed_start);

    // Encode the trimmed segment in base-3, with lower digits assigned to the left side.
    std::uint32_t pattern_code = 0;
    std::uint32_t factor = 1;
    for (int index = trimmed_start; index < trimmed_end; ++index) {
        pattern_code += static_cast<std::uint32_t>(side_cells[static_cast<std::size_t>(index)]) * factor;
        factor *= 3u;
    }
    encoded.pattern_code = pattern_code;

    // Key layout: length bucket, then left-trim count, then base-3 pattern code.
    const std::uint32_t bucket_offset = kLookupBucketOffsets[encoded.effective_length];
    encoded.lookup_key =
        bucket_offset + static_cast<std::uint32_t>(encoded.left_trimmed_empty) * kPow3ByLength[encoded.effective_length] + pattern_code;
    return encoded;
}

EncodedDirectionState decode_direction_lookup_key(std::uint32_t lookup_key) {
    if (lookup_key >= static_cast<std::uint32_t>(kDirectionLookupTableSize)) {
        throw std::out_of_range("direction lookup key out of range");
    }

    EncodedDirectionState decoded{};
    std::uint8_t effective_length = 0;
    // Locate the effective-length bucket first.
    for (std::uint8_t length = 0; length <= kDirectionSideCount; ++length) {
        const std::uint32_t bucket_start = kLookupBucketOffsets[length];
        const std::uint32_t bucket_size = (kDirectionSideCount - length + 1) * kPow3ByLength[length];
        if (lookup_key < bucket_start + bucket_size) {
            effective_length = length;
            break;
        }
    }

    decoded.effective_length = effective_length;
    const std::uint32_t local_key = lookup_key - kLookupBucketOffsets[effective_length];
    const std::uint32_t base = kPow3ByLength[effective_length];
    decoded.left_trimmed_empty = effective_length == 0 ? 0 : static_cast<std::uint8_t>(local_key / base);
    decoded.pattern_code = effective_length == 0 ? 0 : (local_key % base);
    decoded.right_trimmed_empty = static_cast<std::uint8_t>(kDirectionSideCount - effective_length - decoded.left_trimmed_empty);
    decoded.lookup_key = lookup_key;

    // Restore the trimmed segment from the base-3 pattern code.
    std::uint32_t code = decoded.pattern_code;
    for (int index = 0; index < effective_length; ++index) {
        decoded.side_cells[static_cast<std::size_t>(decoded.left_trimmed_empty + index)] =
            static_cast<RelativeCellState>(code % 3u);
        code /= 3u;
    }
    // Rebuild trimmed edges as empty cells for debugging output.
    for (int index = 0; index < decoded.left_trimmed_empty; ++index) {
        decoded.side_cells[static_cast<std::size_t>(index)] = RelativeCellState::kEmpty;
    }
    for (int index = decoded.left_trimmed_empty + effective_length; index < kDirectionSideCount; ++index) {
        decoded.side_cells[static_cast<std::size_t>(index)] = RelativeCellState::kEmpty;
    }

    return decoded;
}

}  // namespace gomoku
