#include "RushFourDataset.h"

#include "../GameStateStore.h"
#include "../RewardEvaluator.h"
#include "../StateValueRegistry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace gomoku::precompute {

namespace {

#define RUSH_FOUR_DATASET_DEBUG 1

constexpr int kBoardSize = 15;
constexpr int kBoardArea = kBoardSize * kBoardSize;
constexpr std::int8_t kEmpty = 0;
constexpr std::int8_t kBlack = 1;
constexpr std::int8_t kWhite = -1;
constexpr const char* kRushFourDebugLogPath = "logs/rush_four_dataset_debug.log";

using FlatBoardArray = GameStateStore::FlatBoardArray;
using FlatEnvState = GameStateStore::FlatEnvState;

struct Coord {
    int row = 0;
    int col = 0;
};

struct ScenarioMeta {
    std::string orientation;
    std::string blocked_side;
    int span_start = 0;
    int span_end = 0;
};

struct Scenario {
    FlatBoardArray board{};
    FlatEnvState state{};
    Coord defense{};
    ScenarioMeta meta{};
    std::vector<int> move_order{};
};

struct OrderedStone {
    int order = 0;
    int player = 0;
    Coord coord{};
};

void append_debug_log_line(const std::string& line) {
#if RUSH_FOUR_DATASET_DEBUG
    std::ofstream output(kRushFourDebugLogPath, std::ios::app);
    if (output.is_open()) {
        output << line << "\n";
    }
#else
    (void)line;
#endif
}

std::string format_dense_counts(const GameStateStore::DenseCountArray& counts) {
    std::ostringstream builder;
    bool first = true;
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (counts[index] == 0) {
            continue;
        }
        if (!first) {
            builder << ", ";
        }
        builder << state_value_name(kTrackedStateValueIds[index]) << "=" << counts[index];
        first = false;
    }
    if (first) {
        return "-";
    }
    return builder.str();
}

std::string format_delta(const RewardEvaluator::DenseCountArray& delta) {
    std::ostringstream builder;
    bool first = true;
    for (std::size_t index = 0; index < delta.size(); ++index) {
        if (delta[index] == 0) {
            continue;
        }
        if (!first) {
            builder << ", ";
        }
        builder << state_value_name(kTrackedStateValueIds[index]) << "=" << delta[index];
        first = false;
    }
    if (first) {
        return "-";
    }
    return builder.str();
}

void reset_debug_log() {
#if RUSH_FOUR_DATASET_DEBUG
    std::ofstream output(kRushFourDebugLogPath, std::ios::trunc);
    if (output.is_open()) {
        output << "[rush_four_dataset_debug]\n";
    }
#endif
}

int tracked_state_index(StateValueId state_id) {
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        if (kTrackedStateValueIds[index] == state_id) {
            return static_cast<int>(index);
        }
    }
    throw std::invalid_argument("state id is not tracked");
}

int action_of(const Coord& coord) {
    return coord.row * kBoardSize + coord.col;
}

bool inside(const Coord& coord) {
    return coord.row >= 0 && coord.row < kBoardSize && coord.col >= 0 && coord.col < kBoardSize;
}

std::int8_t& cell_at(FlatBoardArray* board, const Coord& coord) {
    return board->at(static_cast<std::size_t>(action_of(coord)));
}

std::int8_t cell_at(const FlatBoardArray& board, const Coord& coord) {
    return board[static_cast<std::size_t>(action_of(coord))];
}

Coord rotate_coord_90(const Coord& coord) {
    return Coord{coord.col, kBoardSize - 1 - coord.row};
}

Coord mirror_coord_vertical(const Coord& coord) {
    return Coord{coord.row, kBoardSize - 1 - coord.col};
}

Coord transform_coord(Coord coord, int symmetry_index) {
    const bool mirrored = symmetry_index >= 4;
    int rotations = symmetry_index % 4;
    if (mirrored) {
        coord = mirror_coord_vertical(coord);
    }
    for (int step = 0; step < rotations; ++step) {
        coord = rotate_coord_90(coord);
    }
    return coord;
}

FlatBoardArray transform_board(const FlatBoardArray& board, int symmetry_index) {
    FlatBoardArray transformed{};
    transformed.fill(kEmpty);
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const Coord source{row, col};
            const Coord target = transform_coord(source, symmetry_index);
            transformed[static_cast<std::size_t>(action_of(target))] =
                board[static_cast<std::size_t>(action_of(source))];
        }
    }
    return transformed;
}

FlatBoardArray py_board_to_flat(const py::array_t<std::int8_t>& board_array) {
    FlatBoardArray board{};
    auto view = board_array.unchecked<2>();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            board[static_cast<std::size_t>(action_of(Coord{row, col}))] = view(row, col);
        }
    }
    return board;
}

py::array_t<std::int8_t> flat_to_py_board(const FlatBoardArray& board) {
    py::array_t<std::int8_t> board_array({kBoardSize, kBoardSize});
    auto view = board_array.mutable_unchecked<2>();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            view(row, col) = board[static_cast<std::size_t>(action_of(Coord{row, col}))];
        }
    }
    return board_array;
}

std::string symmetry_name(int symmetry_index) {
    static const std::array<const char*, 8> kNames = {
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "mirror",
        "mirror_rot90",
        "mirror_rot180",
        "mirror_rot270",
    };
    return std::string(kNames[static_cast<std::size_t>(symmetry_index)]);
}

std::vector<OrderedStone> build_move_sequence(const FlatBoardArray& board, std::mt19937* rng) {
    std::vector<OrderedStone> stones;
    stones.reserve(16);
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const Coord coord{row, col};
            const int player = static_cast<int>(cell_at(board, coord));
            if (player == kEmpty) {
                continue;
            }
            stones.push_back(OrderedStone{0, player, coord});
        }
    }
    std::shuffle(stones.begin(), stones.end(), *rng);
    for (std::size_t index = 0; index < stones.size(); ++index) {
        stones[index].order = static_cast<int>(index + 1);
    }
    return stones;
}

FlatEnvState simulate_state_from_sequence(const std::vector<OrderedStone>& stones) {
    constexpr int kDatasetEnvId = -4242;

    GameStateStore& store = GameStateStore::instance();
    RewardEvaluator& evaluator = RewardEvaluator::instance();

    py::array_t<std::int8_t> board = py::array_t<std::int8_t>({kBoardSize, kBoardSize});
    auto board_view = board.mutable_unchecked<2>();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            board_view(row, col) = kEmpty;
        }
    }

    store.resetEnv(kDatasetEnvId, board);
    try {
        append_debug_log_line("[simulate] begin");
        for (const OrderedStone& stone : stones) {
            py::array_t<std::int8_t> board_before = store.board(kDatasetEnvId);
            const auto& before_self = store.countsForPlayer(kDatasetEnvId, stone.player);
            const auto& before_opp = store.countsForPlayer(kDatasetEnvId, -stone.player);
            const RewardResult result = evaluator.evaluate(
                board_before,
                stone.coord.row,
                stone.coord.col,
                stone.player,
                before_self,
                before_opp
            );
            {
                std::ostringstream builder;
                builder
                    << "[step] order=" << stone.order
                    << " player=" << stone.player
                    << " row=" << stone.coord.row
                    << " col=" << stone.coord.col
                    << " self_delta={" << format_delta(result.self_delta) << "}"
                    << " opp_delta={" << format_delta(result.opp_delta) << "}";
                append_debug_log_line(builder.str());
            }
            store.applyMove(kDatasetEnvId, stone.coord.row, stone.coord.col, stone.player);
            {
                const GameStateStore::EnvState& current = store.envState(kDatasetEnvId);
                std::ostringstream builder;
                builder
                    << "[after] black={" << format_dense_counts(current.black_counts) << "}"
                    << " white={" << format_dense_counts(current.white_counts) << "}"
                    << " done=" << current.done
                    << " winner=" << current.winner;
                append_debug_log_line(builder.str());
            }
        }

        const GameStateStore::EnvState& env_state = store.envState(kDatasetEnvId);
        FlatEnvState state{};
        state.board = py_board_to_flat(env_state.board);
        state.black_counts = env_state.black_counts;
        state.white_counts = env_state.white_counts;
        state.done = env_state.done;
        state.winner = env_state.winner;
        {
            std::ostringstream builder;
            builder
                << "[final] black={" << format_dense_counts(state.black_counts) << "}"
                << " white={" << format_dense_counts(state.white_counts) << "}"
                << " done=" << state.done
                << " winner=" << state.winner;
            append_debug_log_line(builder.str());
        }
        store.clearEnv(kDatasetEnvId);
        return state;
    } catch (...) {
        store.clearEnv(kDatasetEnvId);
        throw;
    }
}

std::vector<int> immediate_winning_actions(const FlatBoardArray& board, int player) {
    const int five_index = tracked_state_index(StateValueId::kFive);
    std::vector<int> actions;
    for (int action = 0; action < kBoardArea; ++action) {
        if (board[static_cast<std::size_t>(action)] != kEmpty) {
            continue;
        }
        FlatBoardArray next = board;
        next[static_cast<std::size_t>(action)] = static_cast<std::int8_t>(player);
        const auto counts = GameStateStore::countBoardStatesForPlayer(next, player);
        if (counts[static_cast<std::size_t>(five_index)] > 0) {
            actions.push_back(action);
        }
    }
    return actions;
}

bool is_valid_rush_four_board(const FlatBoardArray& board, int expected_defense_action) {
    const FlatEnvState state = GameStateStore::snapshotFlatBoardState(board);
    if (state.done || state.winner != 0) {
        return false;
    }
    const std::vector<int> wins = immediate_winning_actions(board, kBlack);
    return wins.size() == 1 && wins.front() == expected_defense_action;
}

std::vector<Coord> collect_fill_candidates(const FlatBoardArray& board, int forbidden_action) {
    std::array<bool, kBoardArea> seen{};
    seen.fill(false);
    std::vector<Coord> candidates;
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const Coord center{row, col};
            if (cell_at(board, center) == kEmpty) {
                continue;
            }
            for (int dr = -2; dr <= 2; ++dr) {
                for (int dc = -2; dc <= 2; ++dc) {
                    const Coord candidate{row + dr, col + dc};
                    if (!inside(candidate)) {
                        continue;
                    }
                    const int action = action_of(candidate);
                    if (action == forbidden_action || seen[static_cast<std::size_t>(action)]) {
                        continue;
                    }
                    if (cell_at(board, candidate) != kEmpty) {
                        continue;
                    }
                    seen[static_cast<std::size_t>(action)] = true;
                    candidates.push_back(candidate);
                }
            }
        }
    }
    return candidates;
}

bool place_extra_white_stones(
    FlatBoardArray* board,
    std::mt19937* rng,
    int expected_defense_action,
    int remaining
) {
    if (remaining == 0) {
        return is_valid_rush_four_board(*board, expected_defense_action);
    }

    std::vector<Coord> candidates = collect_fill_candidates(*board, expected_defense_action);
    std::shuffle(candidates.begin(), candidates.end(), *rng);
    for (const Coord& coord : candidates) {
        cell_at(board, coord) = kWhite;
        if (is_valid_rush_four_board(*board, expected_defense_action) &&
            place_extra_white_stones(board, rng, expected_defense_action, remaining - 1)) {
            return true;
        }
        cell_at(board, coord) = kEmpty;
    }
    return false;
}

Scenario build_base_scenario(std::mt19937* rng) {
    std::uniform_int_distribution<int> bool_dist(0, 1);
    const bool diagonal = bool_dist(*rng) == 1;
    const bool use_right_shifted_span = bool_dist(*rng) == 1;
    const bool block_left = bool_dist(*rng) == 1;

    const int start_offset = use_right_shifted_span ? -1 : -2;
    const int end_offset = start_offset + 3;
    const std::string orientation = diagonal ? "diag_main" : "horizontal";
    const std::string blocked_side = block_left ? "left" : "right";

    std::vector<Coord> relative_black;
    for (int offset = start_offset; offset <= end_offset; ++offset) {
        if (diagonal) {
            relative_black.push_back(Coord{offset, offset});
        } else {
            relative_black.push_back(Coord{0, offset});
        }
    }

    const Coord relative_left_end = diagonal ? Coord{start_offset - 1, start_offset - 1} : Coord{0, start_offset - 1};
    const Coord relative_right_end = diagonal ? Coord{end_offset + 1, end_offset + 1} : Coord{0, end_offset + 1};

    int min_row = relative_left_end.row;
    int max_row = relative_right_end.row;
    int min_col = relative_left_end.col;
    int max_col = relative_right_end.col;
    for (const Coord& coord : relative_black) {
        min_row = std::min(min_row, coord.row);
        max_row = std::max(max_row, coord.row);
        min_col = std::min(min_col, coord.col);
        max_col = std::max(max_col, coord.col);
    }

    const int row_shift_min = 2 - min_row;
    const int row_shift_max = (kBoardSize - 3) - max_row;
    const int col_shift_min = 2 - min_col;
    const int col_shift_max = (kBoardSize - 3) - max_col;
    if (row_shift_min > row_shift_max || col_shift_min > col_shift_max) {
        throw std::runtime_error("failed to find legal translation range for rush-four template");
    }

    std::uniform_int_distribution<int> row_dist(row_shift_min, row_shift_max);
    std::uniform_int_distribution<int> col_dist(col_shift_min, col_shift_max);
    const int row_shift = row_dist(*rng);
    const int col_shift = col_dist(*rng);

    const auto translate = [&](const Coord& coord) {
        return Coord{coord.row + row_shift, coord.col + col_shift};
    };

    Scenario scenario{};
    scenario.board.fill(kEmpty);
    scenario.meta.orientation = orientation;
    scenario.meta.blocked_side = blocked_side;
    scenario.meta.span_start = start_offset;
    scenario.meta.span_end = end_offset;

    for (const Coord& coord : relative_black) {
        cell_at(&scenario.board, translate(coord)) = kBlack;
    }

    const Coord blocked = translate(block_left ? relative_left_end : relative_right_end);
    scenario.defense = translate(block_left ? relative_right_end : relative_left_end);
    cell_at(&scenario.board, blocked) = kWhite;

    const int defense_action = action_of(scenario.defense);
    if (!place_extra_white_stones(&scenario.board, rng, defense_action, 2)) {
        throw std::runtime_error("failed to place two white filler stones while preserving a single rush-four defense point");
    }

    if (!is_valid_rush_four_board(scenario.board, defense_action)) {
        throw std::runtime_error("generated board is not a valid single-defense rush-four scenario");
    }

    const std::vector<OrderedStone> stones = build_move_sequence(scenario.board, rng);
    scenario.move_order.assign(kBoardArea, 0);
    for (const OrderedStone& stone : stones) {
        scenario.move_order[static_cast<std::size_t>(action_of(stone.coord))] = stone.order;
    }
    scenario.state = simulate_state_from_sequence(stones);
    return scenario;
}

py::list dense_counts_to_pylist(const GameStateStore::DenseCountArray& counts) {
    py::list payload;
    for (int value : counts) {
        payload.append(value);
    }
    return payload;
}

py::list board_to_pylist(const FlatBoardArray& board) {
    py::list rows;
    for (int row = 0; row < kBoardSize; ++row) {
        py::list cols;
        for (int col = 0; col < kBoardSize; ++col) {
            cols.append(board[static_cast<std::size_t>(row * kBoardSize + col)]);
        }
        rows.append(cols);
    }
    return rows;
}

py::dict scenario_to_pydict(const Scenario& scenario, const std::string& symmetry) {
    py::dict payload;
    payload["symmetry"] = symmetry;
    payload["board"] = board_to_pylist(scenario.board);
    payload["defense_row"] = scenario.defense.row;
    payload["defense_col"] = scenario.defense.col;
    payload["defense_action"] = action_of(scenario.defense);
    payload["black_counts"] = dense_counts_to_pylist(scenario.state.black_counts);
    payload["white_counts"] = dense_counts_to_pylist(scenario.state.white_counts);
    payload["done"] = scenario.state.done;
    payload["winner"] = scenario.state.winner;
    return payload;
}

void write_json_int_array(std::ofstream* output, const GameStateStore::DenseCountArray& counts) {
    *output << "[";
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (index > 0) {
            *output << ", ";
        }
        *output << counts[index];
    }
    *output << "]";
}

void write_json_board(std::ofstream* output, const FlatBoardArray& board) {
    *output << "[";
    for (int row = 0; row < kBoardSize; ++row) {
        if (row > 0) {
            *output << ", ";
        }
        *output << "[";
        for (int col = 0; col < kBoardSize; ++col) {
            if (col > 0) {
                *output << ", ";
            }
            *output << static_cast<int>(board[static_cast<std::size_t>(row * kBoardSize + col)]);
        }
        *output << "]";
    }
    *output << "]";
}

void write_json_scenario(std::ofstream* output, const Scenario& scenario, const std::string& symmetry) {
    *output << "{";
    *output << "\"symmetry\":\"" << symmetry << "\",";
    *output << "\"board\":";
    write_json_board(output, scenario.board);
    *output << ",";
    *output << "\"defense_row\":" << scenario.defense.row << ",";
    *output << "\"defense_col\":" << scenario.defense.col << ",";
    *output << "\"defense_action\":" << action_of(scenario.defense) << ",";
    *output << "\"black_counts\":";
    write_json_int_array(output, scenario.state.black_counts);
    *output << ",";
    *output << "\"white_counts\":";
    write_json_int_array(output, scenario.state.white_counts);
    *output << ",";
    *output << "\"done\":" << (scenario.state.done ? "true" : "false") << ",";
    *output << "\"winner\":" << scenario.state.winner;
    *output << "}";
}

Scenario transform_scenario(const Scenario& source, int symmetry_index) {
    Scenario transformed{};
    transformed.board = transform_board(source.board, symmetry_index);
    transformed.defense = transform_coord(source.defense, symmetry_index);
    transformed.meta = source.meta;
    transformed.move_order.assign(kBoardArea, 0);
    std::vector<OrderedStone> stones;
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const int order = source.move_order[static_cast<std::size_t>(action_of(Coord{row, col}))];
            if (order <= 0) {
                continue;
            }
            const Coord coord = transform_coord(Coord{row, col}, symmetry_index);
            const int player = static_cast<int>(cell_at(transformed.board, coord));
            stones.push_back(OrderedStone{order, player, coord});
            transformed.move_order[static_cast<std::size_t>(action_of(coord))] = order;
        }
    }
    std::sort(stones.begin(), stones.end(), [](const OrderedStone& left, const OrderedStone& right) {
        return left.order < right.order;
    });
    transformed.state = simulate_state_from_sequence(stones);
    return transformed;
}

std::uint32_t normalized_seed(std::uint32_t seed) {
    if (seed != 0) {
        return seed;
    }
    return std::random_device{}();
}

}  // namespace

py::list generate_rush_four_opening_group(std::uint32_t group_count, std::uint32_t seed) {
    if (group_count == 0) {
        throw std::invalid_argument("group_count must be positive");
    }
    reset_debug_log();
    std::mt19937 rng(normalized_seed(seed));

    py::list payload;
    for (std::uint32_t group_index = 0; group_index < group_count; ++group_index) {
        const Scenario base = build_base_scenario(&rng);
        for (int symmetry_index = 0; symmetry_index < 8; ++symmetry_index) {
            const Scenario transformed = transform_scenario(base, symmetry_index);
            py::dict item = scenario_to_pydict(transformed, symmetry_name(symmetry_index));
            item["group_index"] = static_cast<int>(group_index);
            payload.append(item);
        }
    }
    return payload;
}

void write_rush_four_opening_group(const std::string& output_path, std::uint32_t group_count, std::uint32_t seed) {
    if (group_count == 0) {
        throw std::invalid_argument("group_count must be positive");
    }
    reset_debug_log();
    std::mt19937 rng(normalized_seed(seed));

    std::ofstream output(output_path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open output file: " + output_path);
    }

    output << "[";
    bool first_item = true;
    for (std::uint32_t group_index = 0; group_index < group_count; ++group_index) {
        const Scenario base = build_base_scenario(&rng);
        for (int symmetry_index = 0; symmetry_index < 8; ++symmetry_index) {
            if (!first_item) {
                output << ",\n";
            }
            first_item = false;
            const Scenario transformed = transform_scenario(base, symmetry_index);
            output << "{";
            output << "\"group_index\":" << static_cast<int>(group_index) << ",";
            output << "\"symmetry\":\"" << symmetry_name(symmetry_index) << "\",";
            output << "\"board\":";
            write_json_board(&output, transformed.board);
            output << ",";
            output << "\"defense_row\":" << transformed.defense.row << ",";
            output << "\"defense_col\":" << transformed.defense.col << ",";
            output << "\"defense_action\":" << action_of(transformed.defense) << ",";
            output << "\"black_counts\":";
            write_json_int_array(&output, transformed.state.black_counts);
            output << ",";
            output << "\"white_counts\":";
            write_json_int_array(&output, transformed.state.white_counts);
            output << ",";
            output << "\"done\":" << (transformed.state.done ? "true" : "false") << ",";
            output << "\"winner\":" << transformed.state.winner;
            output << "}";
        }
    }
    output << "]\n";
}

}  // namespace gomoku::precompute
