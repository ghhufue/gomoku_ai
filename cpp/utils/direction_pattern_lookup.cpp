#include "direction_pattern_lookup.h"

#include <array>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace gomoku {

namespace {

constexpr int kAnalyzeRadius = kDirectionRadius;
constexpr int kAnalyzeLineLength = kDirectionLineLength;

struct RunSegment {
    int start = 0;
    int end = 0;
    int length = 0;
};

char to_pattern_symbol(RelativeCellState state) {
    switch (state) {
        case RelativeCellState::kEmpty:
            return '_';
        case RelativeCellState::kSelf:
            return 'X';
        case RelativeCellState::kOther:
            return 'O';
        default:
            return '_';
    }
}

std::array<char, kAnalyzeLineLength> rebuild_line(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    RelativeCellState center_state
) {
    if (current_index < 0 || current_index >= kAnalyzeLineLength) {
        throw std::out_of_range("current_index must be inside restored 11-cell line");
    }

    std::array<char, kAnalyzeLineLength> line{};
    int side_index = 0;
    for (int line_index = 0; line_index < kAnalyzeLineLength; ++line_index) {
        if (line_index == current_index) {
            line[static_cast<std::size_t>(line_index)] = to_pattern_symbol(center_state);
            continue;
        }
        line[static_cast<std::size_t>(line_index)] = to_pattern_symbol(side_cells[static_cast<std::size_t>(side_index++)]);
    }
    return line;
}

std::size_t state_value_index(StateValueId state_id) {
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        if (kTrackedStateValueIds[index] == state_id) {
            return index;
        }
    }
    throw std::invalid_argument("state id is not tracked");
}

void increment_count(StateValueCountVector* counts, StateValueId state_id) {
    const std::size_t index = state_value_index(state_id);
    counts->at(index).count += 1;
}

StateValueCountVector make_zero_state_counts() {
    StateValueCountVector counts{};
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        counts[index] = StateValueCount{kTrackedStateValueIds[index], 0};
    }
    return counts;
}

StateValueDeltaVector subtract_counts(const StateValueCountVector& left, const StateValueCountVector& right) {
    StateValueDeltaVector result{};
    for (std::size_t index = 0; index < kTrackedStateValueCount; ++index) {
        result[index] = StateValueDelta{
            kTrackedStateValueIds[index],
            left[index].count - right[index].count,
        };
    }
    return result;
}

std::vector<RunSegment> extract_run_segments(const std::array<char, kAnalyzeLineLength>& line) {
    std::vector<RunSegment> runs;
    for (int index = 0; index < static_cast<int>(line.size()); ++index) {
        if (line[static_cast<std::size_t>(index)] != 'X') {
            continue;
        }
        if (index > 0 && line[static_cast<std::size_t>(index - 1)] == 'X') {
            continue;
        }

        int end = index;
        while (end + 1 < static_cast<int>(line.size()) && line[static_cast<std::size_t>(end + 1)] == 'X') {
            ++end;
        }
        runs.push_back(RunSegment{index, end, end - index + 1});
    }
    return runs;
}

StateValueId contiguous_state_id(int stone_count, bool left_open, bool right_open) {
    const int open_sides = static_cast<int>(left_open) + static_cast<int>(right_open);
    if (stone_count >= 5) {
        return StateValueId::kFive;
    }

    switch (stone_count) {
        case 1:
            if (open_sides == 2) {
                return StateValueId::kLiveOne;
            }
            return open_sides == 1 ? StateValueId::kSleepOne : StateValueId::kDeadOne;
        case 2:
            if (open_sides == 2) {
                return StateValueId::kLiveTwo;
            }
            return open_sides == 1 ? StateValueId::kSleepTwo : StateValueId::kDeadTwo;
        case 3:
            if (open_sides == 2) {
                return StateValueId::kLiveThree;
            }
            return open_sides == 1 ? StateValueId::kSleepThree : StateValueId::kDeadThree;
        case 4:
            if (open_sides == 2) {
                return StateValueId::kLiveFour;
            }
            return open_sides == 1 ? StateValueId::kSleepFour : StateValueId::kDeadFour;
        default:
            throw std::invalid_argument("unsupported contiguous stone count");
    }
}

StateValueId gapped_state_id(int stone_count, int gap_kind, bool left_open, bool right_open) {
    const int open_sides = static_cast<int>(left_open) + static_cast<int>(right_open);

    if (stone_count == 2 && gap_kind == 1) {
        if (open_sides == 2) {
            return StateValueId::kLiveTwoGap1;
        }
        return open_sides == 1 ? StateValueId::kSleepTwoGap1 : StateValueId::kDeadTwoGap1;
    }
    if (stone_count == 3 && gap_kind == 1) {
        if (open_sides == 2) {
            return StateValueId::kLiveThreeGap1;
        }
        return open_sides == 1 ? StateValueId::kSleepThreeGap1 : StateValueId::kDeadThreeGap1;
    }
    if (stone_count == 3 && gap_kind == 2) {
        if (open_sides == 2) {
            return StateValueId::kLiveThreeGap2;
        }
        return open_sides == 1 ? StateValueId::kSleepThreeGap2 : StateValueId::kDeadThreeGap2;
    }
    if (stone_count == 4 && gap_kind == 1) {
        if (open_sides == 2) {
            return StateValueId::kLiveFourGap1;
        }
        return open_sides == 1 ? StateValueId::kSleepFourGap1 : StateValueId::kDeadFourGap1;
    }
    throw std::invalid_argument("unsupported gapped stone count");
}

void accumulate_contiguous_states(const std::array<char, kAnalyzeLineLength>& line, StateValueCountVector* counts) {
    for (const RunSegment& run : extract_run_segments(line)) {
        const bool left_open = run.start > 0 && line[static_cast<std::size_t>(run.start - 1)] == '_';
        const bool right_open = run.end + 1 < static_cast<int>(line.size()) && line[static_cast<std::size_t>(run.end + 1)] == '_';
        increment_count(counts, contiguous_state_id(run.length, left_open, right_open));
    }
}

void accumulate_gapped_states(const std::array<char, kAnalyzeLineLength>& line, StateValueCountVector* counts) {
    const std::vector<RunSegment> runs = extract_run_segments(line);
    for (std::size_t index = 0; index + 1 < runs.size(); ++index) {
        const RunSegment& left = runs[index];
        const RunSegment& right = runs[index + 1];

        bool clear_gap = true;
        for (int pos = left.end + 1; pos < right.start; ++pos) {
            if (line[static_cast<std::size_t>(pos)] != '_') {
                clear_gap = false;
                break;
            }
        }
        if (!clear_gap) {
            continue;
        }

        const int gap_length = right.start - left.end - 1;
        const int stone_count = left.length + right.length;
        int gap_kind = 0;

        if (stone_count == 2 && left.length == 1 && right.length == 1 && gap_length == 1) {
            gap_kind = 1;
        } else if (stone_count == 3 && ((left.length == 2 && right.length == 1) || (left.length == 1 && right.length == 2))) {
            gap_kind = gap_length == 1 ? 1 : 2;
        } else if (stone_count == 4 && gap_length == 1) {
            if (
                (left.length == 3 && right.length == 1) ||
                (left.length == 2 && right.length == 2) ||
                (left.length == 1 && right.length == 3)
            ) {
                gap_kind = 1;
            }
        }

        if (gap_kind == 0) {
            continue;
        }

        const bool left_open = left.start > 0 && line[static_cast<std::size_t>(left.start - 1)] == '_';
        const bool right_open = right.end + 1 < static_cast<int>(line.size()) && line[static_cast<std::size_t>(right.end + 1)] == '_';
        increment_count(counts, gapped_state_id(stone_count, gap_kind, left_open, right_open));
    }
}

}  // namespace

StateValueCountVector classify_restored_direction_state_counts(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    bool fill_current_stone
) {
    return classify_restored_direction_state_counts_with_center(
        side_cells,
        current_index,
        fill_current_stone ? RelativeCellState::kSelf : RelativeCellState::kEmpty
    );
}

StateValueCountVector classify_restored_direction_state_counts_with_center(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    RelativeCellState center_state
) {
    const auto line = rebuild_line(side_cells, current_index, center_state);
    StateValueCountVector counts = make_zero_state_counts();
    accumulate_contiguous_states(line, &counts);
    accumulate_gapped_states(line, &counts);
    return counts;
}

StateValueDeltaVector classify_direction_state_delta(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index
) {
    return classify_direction_state_delta_with_center(side_cells, current_index, RelativeCellState::kSelf);
}

StateValueDeltaVector classify_direction_state_delta_with_center(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    RelativeCellState filled_center_state
) {
    const StateValueCountVector filled_counts =
        classify_restored_direction_state_counts_with_center(side_cells, current_index, filled_center_state);
    const StateValueCountVector empty_counts =
        classify_restored_direction_state_counts_with_center(side_cells, current_index, RelativeCellState::kEmpty);
    return subtract_counts(filled_counts, empty_counts);
}

}  // namespace gomoku
