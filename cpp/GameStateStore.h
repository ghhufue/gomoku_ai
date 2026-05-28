#pragma once

#include "RewardEvaluator.h"
#include "direction_encoding.h"

#include <cstdint>
#include <unordered_map>

#include <pybind11/numpy.h>

namespace gomoku {

class GameStateStore {
public:
    using FlatBoardArray = std::array<std::int8_t, 225>;

    struct EnvState {
        pybind11::array_t<std::int8_t> board{};
        RewardEvaluator::DenseCountArray black_counts{};
        RewardEvaluator::DenseCountArray white_counts{};
        bool done = false;
        int winner = 0;
    };

    static GameStateStore& instance();

    void resetEnv(int env_id, const pybind11::array_t<std::int8_t>& board);
    void clearEnv(int env_id);
    bool hasEnv(int env_id) const;

    const EnvState& envState(int env_id) const;
    const pybind11::array_t<std::int8_t>& board(int env_id) const;
    const RewardEvaluator::DenseCountArray& countsForPlayer(int env_id, int player) const;
    bool isDone(int env_id) const;
    int winner(int env_id) const;

    void applyMove(int env_id, int row, int col, int player);

    static RewardEvaluator::DenseCountArray countBoardStatesForPlayer(
        const pybind11::array_t<std::int8_t>& board_array,
        int player
    );
    static RewardEvaluator::DenseCountArray countBoardStatesForPlayer(
        const FlatBoardArray& board_array,
        int player
    );

private:
    static constexpr int kBoardSize = 15;

    std::unordered_map<int, EnvState> env_states_{};

    static RewardEvaluator::DenseCountArray zeroDenseCounts();
    static std::size_t trackedStateIndex(StateValueId state_id);
    static Stone playerToStone(int player);
    static std::array<RelativeCellState, kDirectionSideCount> extractSideCellsForPlayer(
        const pybind11::array_t<std::int8_t>& board_array,
        int row,
        int col,
        Direction direction,
        int player
    );
    static std::array<RelativeCellState, kDirectionSideCount> extractSideCellsForPlayer(
        const FlatBoardArray& board_array,
        int row,
        int col,
        Direction direction,
        int player
    );
    static void accumulateDenseCounts(
        RewardEvaluator::DenseCountArray* dense_counts,
        const StateValueCountVector& counts
    );
    static void validateBoardShape(const pybind11::array_t<std::int8_t>& board_array);
    static void recomputeState(EnvState* state);
};

bool env_done(int env_id);
int env_winner(int env_id);
void reset_env_state(int env_id, pybind11::array_t<std::int8_t> board);
void clear_env_state(int env_id);
void apply_env_move(int env_id, int row, int col, int player);

}  // namespace gomoku
