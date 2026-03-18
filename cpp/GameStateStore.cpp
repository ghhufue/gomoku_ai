#include "GameStateStore.h"

#include "utils/direction_pattern_lookup.h"

#include <array>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace gomoku {

namespace {

constexpr std::array<Direction, 4> kDirections = {
    Direction::kHorizontal,
    Direction::kVertical,
    Direction::kMainDiagonal,
    Direction::kAntiDiagonal,
};

}  // namespace

GameStateStore& GameStateStore::instance() {
    static GameStateStore store{};
    return store;
}

void GameStateStore::resetEnv(int env_id, const py::array_t<std::int8_t>& board) {
    validateBoardShape(board);
    EnvState state{};
    state.board = py::array_t<std::int8_t>({kBoardSize, kBoardSize});
    auto source = board.unchecked<2>();
    auto target = state.board.mutable_unchecked<2>();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            target(row, col) = source(row, col);
        }
    }
    recomputeState(&state);
    env_states_[env_id] = std::move(state);
}

void GameStateStore::clearEnv(int env_id) {
    env_states_.erase(env_id);
}

bool GameStateStore::hasEnv(int env_id) const {
    return env_states_.find(env_id) != env_states_.end();
}

const GameStateStore::EnvState& GameStateStore::envState(int env_id) const {
    auto it = env_states_.find(env_id);
    if (it == env_states_.end()) {
        throw std::out_of_range("env state is not initialized");
    }
    return it->second;
}

const py::array_t<std::int8_t>& GameStateStore::board(int env_id) const {
    return envState(env_id).board;
}

const RewardEvaluator::DenseCountArray& GameStateStore::countsForPlayer(int env_id, int player) const {
    const EnvState& state = envState(env_id);
    if (player == 1) {
        return state.black_counts;
    }
    if (player == -1) {
        return state.white_counts;
    }
    throw std::invalid_argument("player must be 1 or -1");
}

bool GameStateStore::isDone(int env_id) const {
    return envState(env_id).done;
}

int GameStateStore::winner(int env_id) const {
    return envState(env_id).winner;
}

void GameStateStore::applyMove(int env_id, int row, int col, int player) {
    auto it = env_states_.find(env_id);
    if (it == env_states_.end()) {
        throw std::out_of_range("env state is not initialized");
    }
    if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize) {
        throw std::out_of_range("move coordinate out of range");
    }
    auto board_view = it->second.board.mutable_unchecked<2>();
    if (board_view(row, col) != 0) {
        throw std::invalid_argument("target cell is not empty");
    }
    board_view(row, col) = static_cast<std::int8_t>(player);
    recomputeState(&it->second);
}

RewardEvaluator::DenseCountArray GameStateStore::countBoardStatesForPlayer(
    const py::array_t<std::int8_t>& board_array,
    int player
) {
    validateBoardShape(board_array);
    auto board = board_array.unchecked<2>();
    RewardEvaluator::DenseCountArray totals = zeroDenseCounts();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            if (board(row, col) != player) {
                continue;
            }
            for (Direction direction : kDirections) {
                const auto side_cells = extractSideCellsForPlayer(board_array, row, col, direction, player);
                const StateValueCountVector counts = classify_restored_direction_state_counts_with_center(
                    side_cells,
                    kDirectionRadius,
                    RelativeCellState::kSelf
                );
                accumulateDenseCounts(&totals, counts);
            }
        }
    }
    return totals;
}

RewardEvaluator::DenseCountArray GameStateStore::countBoardStatesForPlayer(
    const FlatBoardArray& board_array,
    int player
) {
    RewardEvaluator::DenseCountArray totals = zeroDenseCounts();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            if (board_array[static_cast<std::size_t>(row * kBoardSize + col)] != player) {
                continue;
            }
            for (Direction direction : kDirections) {
                const auto side_cells = extractSideCellsForPlayer(board_array, row, col, direction, player);
                const StateValueCountVector counts = classify_restored_direction_state_counts_with_center(
                    side_cells,
                    kDirectionRadius,
                    RelativeCellState::kSelf
                );
                accumulateDenseCounts(&totals, counts);
            }
        }
    }
    return totals;
}

RewardEvaluator::DenseCountArray GameStateStore::zeroDenseCounts() {
    RewardEvaluator::DenseCountArray counts{};
    counts.fill(0);
    return counts;
}

std::size_t GameStateStore::trackedStateIndex(StateValueId state_id) {
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        if (kTrackedStateValueIds[index] == state_id) {
            return index;
        }
    }
    throw std::invalid_argument("state id is not tracked");
}

Stone GameStateStore::playerToStone(int player) {
    if (player == 1) {
        return Stone::kBlack;
    }
    if (player == -1) {
        return Stone::kWhite;
    }
    throw std::invalid_argument("player must be 1 or -1");
}

std::array<RelativeCellState, kDirectionSideCount> GameStateStore::extractSideCellsForPlayer(
    const py::array_t<std::int8_t>& board_array,
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
        playerToStone(player),
        kBoardSize
    );
    return line_to_side_cells(line);
}

std::array<RelativeCellState, kDirectionSideCount> GameStateStore::extractSideCellsForPlayer(
    const FlatBoardArray& board_array,
    int row,
    int col,
    Direction direction,
    int player
) {
    const DirectionLine line = extract_direction_line(
        board_array.data(),
        Vec2i{row, col},
        direction,
        playerToStone(player),
        kBoardSize
    );
    return line_to_side_cells(line);
}

void GameStateStore::accumulateDenseCounts(
    RewardEvaluator::DenseCountArray* dense_counts,
    const StateValueCountVector& counts
) {
    for (const StateValueCount& state_count : counts) {
        dense_counts->at(trackedStateIndex(state_count.state_id)) += state_count.count;
    }
}

void GameStateStore::validateBoardShape(const py::array_t<std::int8_t>& board_array) {
    auto board = board_array.unchecked<2>();
    if (board.shape(0) != kBoardSize || board.shape(1) != kBoardSize) {
        throw py::value_error("board must have shape (15, 15)");
    }
}

void GameStateStore::recomputeState(EnvState* state) {
    state->black_counts = countBoardStatesForPlayer(state->board, 1);
    state->white_counts = countBoardStatesForPlayer(state->board, -1);
    const bool black_win = state->black_counts[trackedStateIndex(StateValueId::kFive)] > 0;
    const bool white_win = state->white_counts[trackedStateIndex(StateValueId::kFive)] > 0;
    if (black_win) {
        state->done = true;
        state->winner = 1;
        return;
    }
    if (white_win) {
        state->done = true;
        state->winner = -1;
        return;
    }
    auto board = state->board.unchecked<2>();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            if (board(row, col) == 0) {
                state->done = false;
                state->winner = 0;
                return;
            }
        }
    }
    state->done = true;
    state->winner = 0;
}

bool env_done(int env_id) {
    return GameStateStore::instance().isDone(env_id);
}

int env_winner(int env_id) {
    return GameStateStore::instance().winner(env_id);
}

void reset_env_state(int env_id, py::array_t<std::int8_t> board) {
    GameStateStore::instance().resetEnv(env_id, board);
}

void clear_env_state(int env_id) {
    GameStateStore::instance().clearEnv(env_id);
}

void apply_env_move(int env_id, int row, int col, int player) {
    GameStateStore::instance().applyMove(env_id, row, col, player);
}

}  // namespace gomoku
