#include "GameStateStore.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace gomoku {

namespace {

constexpr int kBoardSizeLocal = 15;

GameStateStore::DenseCountArray denseCountsFromIterable(py::iterable counts_obj) {
    GameStateStore::DenseCountArray counts{};
    counts.fill(0);
    const std::string expected_size = std::to_string(counts.size());
    std::size_t index = 0;
    for (py::handle item : counts_obj) {
        if (index >= counts.size()) {
            throw py::value_error("dense counts must have exactly " + expected_size + " items");
        }
        counts[index++] = py::cast<int>(item);
    }
    if (index != counts.size()) {
        throw py::value_error("dense counts must have exactly " + expected_size + " items");
    }
    return counts;
}

GameStateStore::DenseCountArray zeroDenseCountsLocal() {
    GameStateStore::DenseCountArray counts{};
    counts.fill(0);
    return counts;
}

std::size_t trackedStateIndexLocal(StateValueId state_id) {
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        if (kTrackedStateValueIds[index] == state_id) {
            return index;
        }
    }
    throw std::invalid_argument("state id is not tracked");
}

struct ReplayMove {
    int row = 0;
    int col = 0;
    int player = 0;
};

void copyBoard(
    const py::array_t<std::int8_t>& source_board,
    py::array_t<std::int8_t>* target_board
) {
    auto source = source_board.unchecked<2>();
    auto target = target_board->mutable_unchecked<2>();
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            target(row, col) = source(row, col);
        }
    }
}

py::array_t<std::int8_t> zeroBoard() {
    py::array_t<std::int8_t> board({kBoardSizeLocal, kBoardSizeLocal});
    auto view = board.mutable_unchecked<2>();
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            view(row, col) = 0;
        }
    }
    return board;
}

void addDenseDelta(
    GameStateStore::DenseCountArray* target,
    const RewardEvaluator::DenseCountArray& delta
) {
    for (std::size_t index = 0; index < target->size(); ++index) {
        target->at(index) += delta[index];
    }
}

bool hasEmptyCell(const py::array_t<std::int8_t>& board) {
    auto view = board.unchecked<2>();
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            if (view(row, col) == 0) {
                return true;
            }
        }
    }
    return false;
}

void updateOutcome(
    const py::array_t<std::int8_t>& board,
    const GameStateStore::DenseCountArray& black_counts,
    const GameStateStore::DenseCountArray& white_counts,
    bool* done,
    int* winner
) {
    const std::size_t five_index = trackedStateIndexLocal(StateValueId::kFive);
    if (black_counts[five_index] > 0) {
        *done = true;
        *winner = 1;
        return;
    }
    if (white_counts[five_index] > 0) {
        *done = true;
        *winner = -1;
        return;
    }
    if (hasEmptyCell(board)) {
        *done = false;
        *winner = 0;
        return;
    }
    *done = true;
    *winner = 0;
}

void updateOutcome(
    const GameStateStore::FlatBoardArray& board,
    const GameStateStore::DenseCountArray& black_counts,
    const GameStateStore::DenseCountArray& white_counts,
    bool* done,
    int* winner
) {
    const std::size_t five_index = trackedStateIndexLocal(StateValueId::kFive);
    if (black_counts[five_index] > 0) {
        *done = true;
        *winner = 1;
        return;
    }
    if (white_counts[five_index] > 0) {
        *done = true;
        *winner = -1;
        return;
    }
    for (std::int8_t cell : board) {
        if (cell == 0) {
            *done = false;
            *winner = 0;
            return;
        }
    }
    *done = true;
    *winner = 0;
}

void applyMoveToState(
    GameStateStore::EnvState* state,
    int row,
    int col,
    int player
) {
    RewardEvaluator& evaluator = RewardEvaluator::instance();
    const GameStateStore::DenseCountArray& before_self = (player == 1) ? state->black_counts : state->white_counts;
    const GameStateStore::DenseCountArray& before_opp = (player == 1) ? state->white_counts : state->black_counts;
    RewardResult result = evaluator.evaluate(state->board, row, col, player, before_self, before_opp);

    auto board_view = state->board.mutable_unchecked<2>();
    if (board_view(row, col) != 0) {
        throw std::invalid_argument("target cell is not empty");
    }
    board_view(row, col) = static_cast<std::int8_t>(player);

    if (player == 1) {
        addDenseDelta(&state->black_counts, result.self_delta);
        addDenseDelta(&state->white_counts, result.opp_delta);
    } else {
        addDenseDelta(&state->white_counts, result.self_delta);
        addDenseDelta(&state->black_counts, result.opp_delta);
    }
    updateOutcome(state->board, state->black_counts, state->white_counts, &state->done, &state->winner);
}

std::vector<ReplayMove> collectReplayMoves(const py::array_t<std::int8_t>& board) {
    std::vector<ReplayMove> moves;
    auto view = board.unchecked<2>();
    moves.reserve(static_cast<std::size_t>(kBoardSizeLocal * kBoardSizeLocal));
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            const int player = static_cast<int>(view(row, col));
            if (player == 0) {
                continue;
            }
            moves.push_back(ReplayMove{row, col, player});
        }
    }
    return moves;
}

std::vector<ReplayMove> collectReplayMoves(const GameStateStore::FlatBoardArray& board) {
    std::vector<ReplayMove> moves;
    moves.reserve(board.size());
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            const int player = static_cast<int>(board[static_cast<std::size_t>(row * kBoardSizeLocal + col)]);
            if (player == 0) {
                continue;
            }
            moves.push_back(ReplayMove{row, col, player});
        }
    }
    return moves;
}

GameStateStore::EnvState replayEnvState(const py::array_t<std::int8_t>& board) {
    GameStateStore::EnvState state{};
    state.board = zeroBoard();
    state.black_counts = zeroDenseCountsLocal();
    state.white_counts = zeroDenseCountsLocal();
    state.done = false;
    state.winner = 0;
    for (const ReplayMove& move : collectReplayMoves(board)) {
        applyMoveToState(&state, move.row, move.col, move.player);
    }
    return state;
}

GameStateStore::FlatEnvState replayFlatState(const GameStateStore::FlatBoardArray& board) {
    GameStateStore::EnvState env_state{};
    env_state.board = zeroBoard();
    env_state.black_counts = zeroDenseCountsLocal();
    env_state.white_counts = zeroDenseCountsLocal();
    env_state.done = false;
    env_state.winner = 0;
    for (const ReplayMove& move : collectReplayMoves(board)) {
        applyMoveToState(&env_state, move.row, move.col, move.player);
    }

    GameStateStore::FlatEnvState flat_state{};
    flat_state.black_counts = env_state.black_counts;
    flat_state.white_counts = env_state.white_counts;
    flat_state.done = env_state.done;
    flat_state.winner = env_state.winner;
    auto view = env_state.board.unchecked<2>();
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            flat_state.board[static_cast<std::size_t>(row * kBoardSizeLocal + col)] = view(row, col);
        }
    }
    return flat_state;
}

}  // namespace

GameStateStore& GameStateStore::instance() {
    static GameStateStore store{};
    return store;
}

void GameStateStore::resetEnv(int env_id, const py::array_t<std::int8_t>& board) {
    validateBoardShape(board);
    env_states_[env_id] = replayEnvState(board);
}

void GameStateStore::restoreEnv(
    int env_id,
    const py::array_t<std::int8_t>& board,
    const RewardEvaluator::DenseCountArray& black_counts,
    const RewardEvaluator::DenseCountArray& white_counts,
    bool done,
    int winner
) {
    validateBoardShape(board);
    if (winner != -1 && winner != 0 && winner != 1) {
        throw py::value_error("winner must be -1, 0, or 1");
    }
    if (winner != 0 && !done) {
        throw py::value_error("winner must be 0 when done is false");
    }

    EnvState state{};
    state.board = py::array_t<std::int8_t>({kBoardSize, kBoardSize});
    auto source = board.unchecked<2>();
    auto target = state.board.mutable_unchecked<2>();
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            target(row, col) = source(row, col);
        }
    }
    state.black_counts = black_counts;
    state.white_counts = white_counts;
    state.done = done;
    state.winner = winner;
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
    applyMoveToState(&it->second, row, col, player);
}

GameStateStore::DenseCountArray GameStateStore::countBoardStatesForPlayer(
    const py::array_t<std::int8_t>& board_array,
    int player
) {
    validateBoardShape(board_array);
    const EnvState state = replayEnvState(board_array);
    if (player == 1) {
        return state.black_counts;
    }
    if (player == -1) {
        return state.white_counts;
    }
    throw std::invalid_argument("player must be 1 or -1");
}

GameStateStore::DenseCountArray GameStateStore::countBoardStatesForPlayer(
    const FlatBoardArray& board_array,
    int player
) {
    const FlatEnvState state = replayFlatState(board_array);
    if (player == 1) {
        return state.black_counts;
    }
    if (player == -1) {
        return state.white_counts;
    }
    throw std::invalid_argument("player must be 1 or -1");
}

GameStateStore::FlatEnvState GameStateStore::snapshotFlatBoardState(const FlatBoardArray& board_array) {
    return replayFlatState(board_array);
}

GameStateStore::DenseCountArray GameStateStore::zeroDenseCounts() {
    DenseCountArray counts{};
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

void GameStateStore::validateBoardShape(const py::array_t<std::int8_t>& board_array) {
    auto board = board_array.unchecked<2>();
    if (board.shape(0) != kBoardSize || board.shape(1) != kBoardSize) {
        throw py::value_error("board must have shape (15, 15)");
    }
}

bool env_done(int env_id) {
    return GameStateStore::instance().isDone(env_id);
}

int env_winner(int env_id) {
    return GameStateStore::instance().winner(env_id);
}

int infer_next_player(py::array_t<std::int8_t> board) {
    auto view = board.unchecked<2>();
    if (view.shape(0) != kBoardSizeLocal || view.shape(1) != kBoardSizeLocal) {
        throw py::value_error("board must have shape (15, 15)");
    }
    int black_stones = 0;
    int white_stones = 0;
    for (int row = 0; row < kBoardSizeLocal; ++row) {
        for (int col = 0; col < kBoardSizeLocal; ++col) {
            const int value = static_cast<int>(view(row, col));
            if (value == 1) {
                ++black_stones;
            } else if (value == -1) {
                ++white_stones;
            } else if (value != 0) {
                throw py::value_error("board contains invalid stone values");
            }
        }
    }
    if (black_stones == white_stones) {
        return 1;
    }
    if (black_stones == white_stones + 1) {
        return -1;
    }
    throw py::value_error("board has invalid black/white stone counts");
}

void reset_env_state(int env_id, py::array_t<std::int8_t> board) {
    GameStateStore::instance().resetEnv(env_id, board);
}

void restore_env_state(
    int env_id,
    py::array_t<std::int8_t> board,
    py::iterable black_counts,
    py::iterable white_counts,
    bool done,
    int winner
) {
    GameStateStore::instance().restoreEnv(
        env_id,
        board,
        denseCountsFromIterable(black_counts),
        denseCountsFromIterable(white_counts),
        done,
        winner
    );
}

void clear_env_state(int env_id) {
    GameStateStore::instance().clearEnv(env_id);
}

void apply_env_move(int env_id, int row, int col, int player) {
    GameStateStore::instance().applyMove(env_id, row, col, player);
}

}  // namespace gomoku
