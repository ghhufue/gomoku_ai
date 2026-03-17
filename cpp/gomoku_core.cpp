#include "direction_encoding.h"
#include "precompute/direction_delta_table.h"
#include "reward_config.h"
#include "state_value.h"
#include "utils/direction_pattern_lookup.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace gomoku {

namespace {

constexpr int kBoardSize = 15;
constexpr int kEmpty = 0;
constexpr int kOpponentEventOffset = 100000;
constexpr std::array<Direction, 4> kDirections = {
    Direction::kHorizontal,
    Direction::kVertical,
    Direction::kMainDiagonal,
    Direction::kAntiDiagonal,
};

struct RewardResult {
    double reward = 0.0;
    double offense_score = 0.0;
    double defense_score = 0.0;
    std::vector<std::pair<int, int>> events{};
};

Stone player_to_stone(int player) {
    if (player == 1) {
        return Stone::kBlack;
    }
    if (player == -1) {
        return Stone::kWhite;
    }
    throw std::invalid_argument("player must be 1 or -1");
}

using DenseCountArray = std::array<int, kTrackedStateValueCount>;

DenseCountArray zero_dense_counts() {
    DenseCountArray counts{};
    counts.fill(0);
    return counts;
}

std::size_t tracked_state_index(StateValueId state_id) {
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        if (kTrackedStateValueIds[index] == state_id) {
            return index;
        }
    }
    throw std::invalid_argument("state id is not tracked");
}

std::array<RelativeCellState, kDirectionSideCount> extract_side_cells_for_player(
    py::array_t<std::int8_t>& board_array,
    int row,
    int col,
    Direction direction,
    int player
) {
    auto board = board_array.unchecked<2>();
    const DirectionLine line = extract_direction_line(
        board.data(0, 0),
        Vec2i{row, col},
        direction,
        player_to_stone(player),
        kBoardSize
    );
    return line_to_side_cells(line);
}

void accumulate_dense_delta(
    DenseCountArray* dense_delta,
    const std::array<std::int8_t, kTrackedStateValueCount>& packed_delta
) {
    for (std::size_t index = 0; index < kTrackedStateValueCount; ++index) {
        dense_delta->at(index) += static_cast<int>(packed_delta[index]);
    }
}

void accumulate_dense_counts(
    DenseCountArray* dense_counts,
    const StateValueCountVector& counts
) {
    for (const StateValueCount& state_count : counts) {
        dense_counts->at(tracked_state_index(state_count.state_id)) += state_count.count;
    }
}

DenseCountArray apply_delta(const DenseCountArray& previous, const DenseCountArray& delta) {
    DenseCountArray current = previous;
    for (std::size_t index = 0; index < current.size(); ++index) {
        current[index] += delta[index];
        if (current[index] < 0) {
            throw std::runtime_error("state count became negative after applying local delta");
        }
    }
    return current;
}

std::vector<std::pair<int, int>> build_event_pairs(
    const DenseCountArray& self_delta,
    const DenseCountArray& opp_delta
) {
    std::vector<std::pair<int, int>> events;
    for (std::size_t index = 0; index < kTrackedStateValueCount; ++index) {
        if (self_delta[index] != 0) {
            events.emplace_back(static_cast<int>(kTrackedStateValueIds[index]), self_delta[index]);
        }
        if (opp_delta[index] != 0) {
            events.emplace_back(kOpponentEventOffset + static_cast<int>(kTrackedStateValueIds[index]), opp_delta[index]);
        }
    }
    return events;
}

double score_delta(const DenseCountArray& delta, const RewardConfig& config, bool invert_sign) {
    double score = 0.0;
    for (std::size_t index = 0; index < kTrackedStateValueCount; ++index) {
        const int signed_delta = invert_sign ? -delta[index] : delta[index];
        score += static_cast<double>(signed_delta) * config.state_scores[index];
    }
    return score;
}

int sum_states(const DenseCountArray& counts, std::initializer_list<StateValueId> state_ids) {
    int total = 0;
    for (StateValueId state_id : state_ids) {
        total += counts[tracked_state_index(state_id)];
    }
    return total;
}

int live_three_total(const DenseCountArray& counts) {
    return sum_states(counts, {
        StateValueId::kLiveThree,
        StateValueId::kLiveThreeGap1,
        StateValueId::kLiveThreeGap2,
    });
}

int four_threat_total(const DenseCountArray& counts) {
    return sum_states(counts, {
        StateValueId::kLiveFour,
        StateValueId::kLiveFourGap1,
        StateValueId::kSleepFour,
        StateValueId::kSleepFourGap1,
    });
}

int winning_total(const DenseCountArray& counts) {
    return sum_states(counts, {StateValueId::kFive});
}

DenseCountArray count_board_states_for_player(py::array_t<std::int8_t>& board_array, int player) {
    auto board = board_array.unchecked<2>();
    DenseCountArray totals = zero_dense_counts();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            if (board(row, col) != player) {
                continue;
            }
            for (Direction direction : kDirections) {
                const auto side_cells = extract_side_cells_for_player(board_array, row, col, direction, player);
                const StateValueCountVector counts = classify_restored_direction_state_counts_with_center(
                    side_cells,
                    kDirectionRadius,
                    RelativeCellState::kSelf
                );
                accumulate_dense_counts(&totals, counts);
            }
        }
    }
    return totals;
}

py::array_t<std::int8_t> build_board_after(
    py::array_t<std::int8_t>& board_before,
    int row,
    int col,
    int player
) {
    py::array_t<std::int8_t> board_after({kBoardSize, kBoardSize});
    auto before = board_before.unchecked<2>();
    auto after = board_after.mutable_unchecked<2>();
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            after(r, c) = before(r, c);
        }
    }
    after(row, col) = static_cast<std::int8_t>(player);
    return board_after;
}

RewardResult evaluate_reward_impl(
    py::array_t<std::int8_t>& board_before,
    int row,
    int col,
    int player
) {
    auto board = board_before.unchecked<2>();
    if (board.shape(0) != kBoardSize || board.shape(1) != kBoardSize) {
        throw py::value_error("board_before must have shape (15, 15)");
    }
    if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize) {
        throw py::value_error("move coordinate out of range");
    }
    if (board(row, col) != kEmpty) {
        throw py::value_error("board_before[row, col] must be empty");
    }

    DenseCountArray self_delta = zero_dense_counts();
    DenseCountArray opp_delta = zero_dense_counts();
    for (Direction direction : kDirections) {
        const auto self_side_cells = extract_side_cells_for_player(board_before, row, col, direction, player);
        const PackedDirectionDeltaEntry& self_entry = lookup_direction_delta_entry(self_side_cells);
        accumulate_dense_delta(&self_delta, self_entry.self_delta);

        const auto opp_side_cells = extract_side_cells_for_player(board_before, row, col, direction, -player);
        const PackedDirectionDeltaEntry& opp_entry = lookup_direction_delta_entry(opp_side_cells);
        accumulate_dense_delta(&opp_delta, opp_entry.other_delta);
    }

    const RewardConfig& config = current_reward_config();
    const DenseCountArray before_self = count_board_states_for_player(board_before, player);
    const DenseCountArray before_opp = count_board_states_for_player(board_before, -player);
    py::array_t<std::int8_t> board_after = build_board_after(board_before, row, col, player);
    const DenseCountArray after_self = count_board_states_for_player(board_after, player);
    const DenseCountArray after_opp = count_board_states_for_player(board_after, -player);

    RewardResult result{};
    result.offense_score = score_delta(self_delta, config, false);
    result.defense_score = score_delta(opp_delta, config, true);
    result.reward = config.step_penalty + result.offense_score + result.defense_score;

    if (four_threat_total(before_self) > 0 && winning_total(after_self) == 0) {
        result.reward -= config.missed_immediate_win_penalty;
    }

    if (live_three_total(self_delta) >= 2) {
        result.reward += config.double_live_three_bonus;
    }

    if (winning_total(before_opp) > 0 && winning_total(after_opp) == 0) {
        result.reward += config.block_winning_bonus;
    } else if (four_threat_total(before_opp) > four_threat_total(after_opp)) {
        result.reward += config.block_live_four_bonus;
    } else if (live_three_total(before_opp) > live_three_total(after_opp)) {
        result.reward += config.block_live_three_bonus;
    }

    if (winning_total(before_opp) > 0 && winning_total(after_opp) > 0) {
        result.reward -= config.unresolved_winning_threat_penalty;
    } else if (four_threat_total(before_opp) > 0 && four_threat_total(after_opp) >= four_threat_total(before_opp)) {
        result.reward -= config.unresolved_four_threat_penalty;
    } else if (
        live_three_total(before_opp) > 0 &&
        live_three_total(after_opp) >= live_three_total(before_opp) &&
        four_threat_total(after_self) == 0
    ) {
        result.reward -= config.unresolved_live_three_fatal_penalty;
    }

    result.events = build_event_pairs(self_delta, opp_delta);
    return result;
}

py::list list_state_values() {
    py::list payload;
    for (const StateValueDef& def : state_value_definitions()) {
        py::dict item;
        item["id"] = static_cast<int>(def.state_id);
        item["name"] = py::str(def.name);
        item["description"] = py::str(def.description);
        payload.append(item);
    }
    return payload;
}

py::dict debug_encode_direction_side_states(py::iterable states_obj) {
    std::array<RelativeCellState, kDirectionSideCount> states{};
    int index = 0;
    for (py::handle item : states_obj) {
        if (index >= kDirectionSideCount) {
            throw py::value_error("expected at most 10 side states");
        }
        const int value = py::cast<int>(item);
        if (value < 0 || value > 2) {
            throw py::value_error("side state must be 0, 1, or 2");
        }
        states[static_cast<std::size_t>(index++)] = static_cast<RelativeCellState>(value);
    }
    if (index != kDirectionSideCount) {
        throw py::value_error("expected exactly 10 side states");
    }

    const EncodedDirectionState encoded = encode_direction_side_cells(states);
    py::dict payload;
    payload["effective_length"] = encoded.effective_length;
    payload["left_trimmed_empty"] = encoded.left_trimmed_empty;
    payload["right_trimmed_empty"] = encoded.right_trimmed_empty;
    payload["pattern_code"] = encoded.pattern_code;
    payload["lookup_key"] = encoded.lookup_key;
    return payload;
}

py::dict debug_decode_direction_lookup_key(std::uint32_t lookup_key) {
    const EncodedDirectionState decoded = decode_direction_lookup_key(lookup_key);
    py::list states;
    for (RelativeCellState state : decoded.side_cells) {
        states.append(static_cast<int>(state));
    }

    py::dict payload;
    payload["effective_length"] = decoded.effective_length;
    payload["left_trimmed_empty"] = decoded.left_trimmed_empty;
    payload["right_trimmed_empty"] = decoded.right_trimmed_empty;
    payload["pattern_code"] = decoded.pattern_code;
    payload["lookup_key"] = decoded.lookup_key;
    payload["side_states"] = states;
    return payload;
}

py::list debug_classify_direction_side_states(py::iterable states_obj) {
    std::array<RelativeCellState, kDirectionSideCount> states{};
    int index = 0;
    for (py::handle item : states_obj) {
        if (index >= kDirectionSideCount) {
            throw py::value_error("expected at most 10 side states");
        }
        const int value = py::cast<int>(item);
        if (value < 0 || value > 2) {
            throw py::value_error("side state must be 0, 1, or 2");
        }
        states[static_cast<std::size_t>(index++)] = static_cast<RelativeCellState>(value);
    }
    if (index != kDirectionSideCount) {
        throw py::value_error("expected exactly 10 side states");
    }

    const StateValueCountVector counts = classify_restored_direction_state_counts(states, kDirectionRadius, true);
    py::list payload;
    for (const StateValueCount& state_count : counts) {
        if (state_count.count > 0) {
            payload.append(py::str(state_value_name(state_count.state_id)));
        }
    }
    return payload;
}

py::dict evaluate_reward(
    py::array_t<std::int8_t> board_before,
    int row,
    int col,
    int player
) {
    RewardResult result = evaluate_reward_impl(board_before, row, col, player);

    py::dict payload;
    payload["reward"] = result.reward;
    payload["offense_score"] = result.offense_score;
    payload["defense_score"] = result.defense_score;
    payload["events"] = py::cast(result.events);
    return payload;
}

}  // namespace

}  // namespace gomoku

PYBIND11_MODULE(_cpp_backend, module) {
    module.doc() = "C++ backend for Gomoku reward evaluation";
    module.def(
        "evaluate_reward",
        &gomoku::evaluate_reward,
        py::arg("board_before"),
        py::arg("row"),
        py::arg("col"),
        py::arg("player")
    );
    module.def("list_state_values", &gomoku::list_state_values);
    module.def("debug_encode_direction_side_states", &gomoku::debug_encode_direction_side_states, py::arg("states"));
    module.def("debug_decode_direction_lookup_key", &gomoku::debug_decode_direction_lookup_key, py::arg("lookup_key"));
    module.def("debug_classify_direction_side_states", &gomoku::debug_classify_direction_side_states, py::arg("states"));
}
