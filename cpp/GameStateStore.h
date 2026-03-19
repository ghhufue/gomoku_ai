#pragma once

#include "RewardEvaluator.h"

#include <cstdint>
#include <unordered_map>

#include <pybind11/numpy.h>

namespace gomoku {

class GameStateStore {
public:
    using FlatBoardArray = std::array<std::int8_t, 225>;
    using DenseCountArray = RewardEvaluator::DenseCountArray;

    struct EnvState {
        pybind11::array_t<std::int8_t> board{};
        DenseCountArray black_counts{};
        DenseCountArray white_counts{};
        bool done = false;
        int winner = 0;
    };

    struct FlatEnvState {
        FlatBoardArray board{};
        DenseCountArray black_counts{};
        DenseCountArray white_counts{};
        bool done = false;
        int winner = 0;
    };

    static GameStateStore& instance();

    void resetEnv(int env_id, const pybind11::array_t<std::int8_t>& board);
    void restoreEnv(
        int env_id,
        const pybind11::array_t<std::int8_t>& board,
        const RewardEvaluator::DenseCountArray& black_counts,
        const RewardEvaluator::DenseCountArray& white_counts,
        bool done,
        int winner
    );
    void clearEnv(int env_id);
    bool hasEnv(int env_id) const;

    const EnvState& envState(int env_id) const;
    const pybind11::array_t<std::int8_t>& board(int env_id) const;
    const RewardEvaluator::DenseCountArray& countsForPlayer(int env_id, int player) const;
    bool isDone(int env_id) const;
    int winner(int env_id) const;

    void applyMove(int env_id, int row, int col, int player);

    static DenseCountArray countBoardStatesForPlayer(
        const pybind11::array_t<std::int8_t>& board_array,
        int player
    );
    static DenseCountArray countBoardStatesForPlayer(
        const FlatBoardArray& board_array,
        int player
    );
    static FlatEnvState snapshotFlatBoardState(const FlatBoardArray& board_array);

private:
    static constexpr int kBoardSize = 15;

    std::unordered_map<int, EnvState> env_states_{};

    static DenseCountArray zeroDenseCounts();
    static std::size_t trackedStateIndex(StateValueId state_id);
    static void validateBoardShape(const pybind11::array_t<std::int8_t>& board_array);
};

bool env_done(int env_id);
int env_winner(int env_id);
int infer_next_player(pybind11::array_t<std::int8_t> board);
void reset_env_state(int env_id, pybind11::array_t<std::int8_t> board);
void restore_env_state(
    int env_id,
    pybind11::array_t<std::int8_t> board,
    pybind11::iterable black_counts,
    pybind11::iterable white_counts,
    bool done,
    int winner
);
void clear_env_state(int env_id);
void apply_env_move(int env_id, int row, int col, int player);

}  // namespace gomoku
