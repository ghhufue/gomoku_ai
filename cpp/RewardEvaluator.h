#pragma once

#include "RewardConfigStore.h"
#include "StateValueRegistry.h"
#include "direction_encoding.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace gomoku {

struct RewardResult {
    double reward = 0.0;
    double offense_score = 0.0;
    double defense_score = 0.0;
    double step_penalty = 0.0;
    double double_live_three_bonus = 0.0;
    double block_winning_bonus = 0.0;
    double block_live_four_bonus = 0.0;
    double block_live_three_bonus = 0.0;
    double unresolved_winning_threat_penalty = 0.0;
    double unresolved_four_threat_penalty = 0.0;
    double unresolved_live_three_fatal_penalty = 0.0;
    double missed_immediate_win_penalty = 0.0;
    std::vector<std::pair<int, int>> events{};
    std::array<int, kTrackedStateValueCount> self_delta{};
    std::array<int, kTrackedStateValueCount> opp_delta{};
};

class RewardEvaluator {
public:
    using DenseCountArray = std::array<int, kTrackedStateValueCount>;

    static RewardEvaluator& instance();

    RewardResult evaluate(
        pybind11::array_t<std::int8_t>& board_before,
        int row,
        int col,
        int player,
        const DenseCountArray& before_self,
        const DenseCountArray& before_opp
    ) const;

private:
    static Stone playerToStone(int player);
    static DenseCountArray zeroDenseCounts();
    static std::size_t trackedStateIndex(StateValueId state_id);
    static std::array<RelativeCellState, kDirectionSideCount> extractSideCellsForPlayer(
        pybind11::array_t<std::int8_t>& board_array,
        int row,
        int col,
        Direction direction,
        int player
    );
    static void accumulateDenseDelta(
        DenseCountArray* dense_delta,
        const std::array<std::int8_t, kTrackedStateValueCount>& packed_delta
    );
    static void accumulateDenseCounts(DenseCountArray* dense_counts, const StateValueCountVector& counts);
    static std::vector<std::pair<int, int>> buildEventPairs(
        const DenseCountArray& self_delta,
        const DenseCountArray& opp_delta
    );
    static double scoreDelta(const DenseCountArray& delta, const RewardConfig& config, bool invert_sign);
    static int sumStates(const DenseCountArray& counts, std::initializer_list<StateValueId> state_ids);
    static int liveThreeTotal(const DenseCountArray& counts);
    static int fourThreatTotal(const DenseCountArray& counts);
    static int winningTotal(const DenseCountArray& counts);
    static DenseCountArray applyDelta(const DenseCountArray& previous, const DenseCountArray& delta);
    static pybind11::array_t<std::int8_t> copyBoardAndApplyMove(
        const pybind11::array_t<std::int8_t>& board_before,
        int row,
        int col,
        int player
    );
};

pybind11::list list_state_values();
pybind11::list decode_reward_events(pybind11::iterable events_obj);
pybind11::dict debug_encode_direction_side_states(pybind11::iterable states_obj);
pybind11::dict debug_decode_direction_lookup_key(std::uint32_t lookup_key);
pybind11::list debug_classify_direction_side_states(pybind11::iterable states_obj);
pybind11::dict evaluate_reward(pybind11::array_t<std::int8_t> board_before, int row, int col, int player);
pybind11::dict evaluate_env_reward(int env_id, int row, int col, int player);
pybind11::list score_classic_candidates(
    pybind11::array_t<std::int8_t> board_before,
    pybind11::iterable actions_obj,
    int player,
    int thread_batch_size = 10
);
pybind11::list score_reward_candidates(
    pybind11::array_t<std::int8_t> board_before,
    pybind11::iterable actions_obj,
    int player,
    int thread_batch_size = 10
);

}  // namespace gomoku
