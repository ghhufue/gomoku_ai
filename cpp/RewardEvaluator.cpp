#include "RewardEvaluator.h"
#include "GameStateStore.h"
#include "precompute/DirectionDeltaTable.h"
#include "RewardConfigStore.h"
#include "utils/direction_pattern_lookup.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace gomoku
{

    namespace
    {

        constexpr int kBoardSize = 15;
        constexpr int kBoardArea = kBoardSize * kBoardSize;
        constexpr int kEmpty = 0;
        constexpr int kOpponentEventOffset = 100000;
        constexpr std::array<Direction, 4> kDirections = {
            Direction::kHorizontal,
            Direction::kVertical,
            Direction::kMainDiagonal,
            Direction::kAntiDiagonal,
        };

        struct ClassicCandidateScore
        {
            int action = -1;
            double score = -std::numeric_limits<double>::infinity();
            bool self_win = false;
            bool opp_win = false;
        };

        struct RewardCandidateScore
        {
            int action = -1;
            double reward = -std::numeric_limits<double>::infinity();
            double offense_score = 0.0;
            double defense_score = 0.0;
        };

        using FlatBoardArray = GameStateStore::FlatBoardArray;

        Stone player_to_stone_local(int player)
        {
            if (player == 1)
            {
                return Stone::kBlack;
            }
            if (player == -1)
            {
                return Stone::kWhite;
            }
            throw std::invalid_argument("player must be 1 or -1");
        }

        RewardEvaluator::DenseCountArray zero_dense_counts_local()
        {
            RewardEvaluator::DenseCountArray counts{};
            counts.fill(0);
            return counts;
        }

        void accumulate_dense_delta_local(
            RewardEvaluator::DenseCountArray *dense_delta,
            const std::array<std::int8_t, kTrackedStateValueCount> &packed_delta)
        {
            for (std::size_t index = 0; index < kTrackedStateValueCount; ++index)
            {
                dense_delta->at(index) += static_cast<int>(packed_delta[index]);
            }
        }

        std::size_t tracked_state_index_local(StateValueId state_id)
        {
            for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index)
            {
                if (kTrackedStateValueIds[index] == state_id)
                {
                    return index;
                }
            }
            throw std::invalid_argument("state id is not tracked");
        }

        int sum_states_local(
            const RewardEvaluator::DenseCountArray &counts,
            std::initializer_list<StateValueId> state_ids)
        {
            int total = 0;
            for (StateValueId state_id : state_ids)
            {
                total += counts[tracked_state_index_local(state_id)];
            }
            return total;
        }

        int live_three_total_local(const RewardEvaluator::DenseCountArray &counts)
        {
            return sum_states_local(counts, {
                                                StateValueId::kLiveThree,
                                                StateValueId::kLiveThreeGap1,
                                                StateValueId::kLiveThreeGap2,
                                            });
        }

        int four_threat_total_local(const RewardEvaluator::DenseCountArray &counts)
        {
            return sum_states_local(counts, {
                                                StateValueId::kLiveFour,
                                                StateValueId::kLiveFourGap1,
                                                StateValueId::kSleepFour,
                                                StateValueId::kSleepFourGap1,
                                            });
        }

        int winning_total_local(const RewardEvaluator::DenseCountArray &counts)
        {
            return sum_states_local(counts, {StateValueId::kFive});
        }

        double score_delta_local(
            const RewardEvaluator::DenseCountArray &delta,
            const RewardConfig &config,
            bool invert_sign)
        {
            double score = 0.0;
            for (std::size_t index = 0; index < kTrackedStateValueCount; ++index)
            {
                const int signed_delta = invert_sign ? -delta[index] : delta[index];
                score += static_cast<double>(signed_delta) * config.state_scores[index];
            }
            return score;
        }

        py::dict build_special_rewards_payload(const RewardResult &result)
        {
            py::dict payload;
            payload["step_penalty"] = result.step_penalty;
            payload["double_live_three_bonus"] = result.double_live_three_bonus;
            payload["block_winning_bonus"] = result.block_winning_bonus;
            payload["block_live_four_bonus"] = result.block_live_four_bonus;
            payload["block_live_three_bonus"] = result.block_live_three_bonus;
            payload["unresolved_winning_threat_penalty"] = result.unresolved_winning_threat_penalty;
            payload["unresolved_four_threat_penalty"] = result.unresolved_four_threat_penalty;
            payload["unresolved_live_three_fatal_penalty"] = result.unresolved_live_three_fatal_penalty;
            payload["missed_immediate_win_penalty"] = result.missed_immediate_win_penalty;
            return payload;
        }

        std::array<RelativeCellState, kDirectionSideCount> extract_side_cells_from_flat_board(
            const FlatBoardArray &board,
            int row,
            int col,
            Direction direction,
            int player)
        {
            const DirectionLine line = extract_direction_line(
                board.data(),
                Vec2i{row, col},
                direction,
                player_to_stone_local(player),
                kBoardSize);
            return line_to_side_cells(line);
        }

        RewardEvaluator::DenseCountArray compute_self_delta_for_move(
            const FlatBoardArray &board,
            int row,
            int col,
            int player)
        {
            RewardEvaluator::DenseCountArray delta = zero_dense_counts_local();
            for (Direction direction : kDirections)
            {
                const auto side_cells = extract_side_cells_from_flat_board(board, row, col, direction, player);
                const PackedDirectionDeltaEntry &entry = lookup_direction_delta_entry(side_cells);
                accumulate_dense_delta_local(&delta, entry.self_delta);
            }
            return delta;
        }

        double classic_state_score(const RewardEvaluator::DenseCountArray &delta)
        {
            const auto sum_group = [&](std::initializer_list<StateValueId> ids)
            {
                int total = 0;
                for (StateValueId state_id : ids)
                {
                    total += delta[tracked_state_index_local(state_id)];
                }
                return total;
            };

            double score = 0.0;
            score += static_cast<double>(sum_group({
                         StateValueId::kLiveFour,
                         StateValueId::kLiveFourGap1,
                     })) *
                     600.0;
            score += static_cast<double>(sum_group({
                         StateValueId::kSleepFour,
                         StateValueId::kSleepFourGap1,
                     })) *
                     260.0;
            score += static_cast<double>(sum_group({
                         StateValueId::kLiveThree,
                         StateValueId::kLiveThreeGap1,
                         StateValueId::kLiveThreeGap2,
                     })) *
                     90.0;
            score += static_cast<double>(sum_group({
                         StateValueId::kSleepThree,
                         StateValueId::kSleepThreeGap1,
                         StateValueId::kSleepThreeGap2,
                     })) *
                     30.0;
            score += static_cast<double>(sum_group({
                         StateValueId::kLiveTwo,
                         StateValueId::kLiveTwoGap1,
                     })) *
                     12.0;
            return score;
        }

        FlatBoardArray copy_flat_board(const py::array_t<std::int8_t> &board_before)
        {
            auto board = board_before.unchecked<2>();
            if (board.shape(0) != kBoardSize || board.shape(1) != kBoardSize)
            {
                throw py::value_error("board_before must have shape (15, 15)");
            }

            FlatBoardArray flat{};
            for (int row = 0; row < kBoardSize; ++row)
            {
                for (int col = 0; col < kBoardSize; ++col)
                {
                    flat[static_cast<std::size_t>(row * kBoardSize + col)] = board(row, col);
                }
            }
            return flat;
        }

    } // namespace

    RewardEvaluator &RewardEvaluator::instance()
    {
        static RewardEvaluator evaluator{};
        return evaluator;
    }

    RewardResult RewardEvaluator::evaluate(
        py::array_t<std::int8_t> &board_before,
        int row,
        int col,
        int player,
        const DenseCountArray &before_self,
        const DenseCountArray &before_opp) const
    {
        auto board = board_before.unchecked<2>();
        if (board.shape(0) != kBoardSize || board.shape(1) != kBoardSize)
        {
            throw py::value_error("board_before must have shape (15, 15)");
        }
        if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize)
        {
            throw py::value_error("move coordinate out of range");
        }
        if (board(row, col) != kEmpty)
        {
            throw py::value_error("board_before[row, col] must be empty");
        }

        DenseCountArray self_delta = zeroDenseCounts();
        DenseCountArray opp_delta = zeroDenseCounts();
        for (Direction direction : kDirections)
        {
            const auto self_side_cells = extractSideCellsForPlayer(board_before, row, col, direction, player);
            const PackedDirectionDeltaEntry &self_entry = lookup_direction_delta_entry(self_side_cells);
            accumulateDenseDelta(&self_delta, self_entry.self_delta);

            const auto opp_side_cells = extractSideCellsForPlayer(board_before, row, col, direction, -player);
            const PackedDirectionDeltaEntry &opp_entry = lookup_direction_delta_entry(opp_side_cells);
            accumulateDenseDelta(&opp_delta, opp_entry.other_delta);
        }

        const RewardConfig &config = current_reward_config();
        py::array_t<std::int8_t> board_after = copyBoardAndApplyMove(board_before, row, col, player);
        const DenseCountArray after_self = GameStateStore::countBoardStatesForPlayer(board_after, player);
        const DenseCountArray after_opp = GameStateStore::countBoardStatesForPlayer(board_after, -player);

        RewardResult result{};
        result.offense_score = scoreDelta(self_delta, config, false);
        result.defense_score = scoreDelta(opp_delta, config, true);
        result.step_penalty = config.step_penalty;
        result.reward = result.step_penalty + result.offense_score + result.defense_score;
        result.self_delta = self_delta;
        result.opp_delta = opp_delta;

        if (fourThreatTotal(before_self) > 0 && winningTotal(after_self) == 0)
        {
            result.missed_immediate_win_penalty = -config.missed_immediate_win_penalty;
            result.reward += result.missed_immediate_win_penalty;
        }
        if (liveThreeTotal(self_delta) >= 2)
        {
            result.double_live_three_bonus = config.double_live_three_bonus;
            result.reward += result.double_live_three_bonus;
        }
        if (winningTotal(before_opp) > 0 && winningTotal(after_opp) == 0)
        {
            result.block_winning_bonus = config.block_winning_bonus;
            result.reward += result.block_winning_bonus;
        }
        else if (liveThreeTotal(before_opp) > liveThreeTotal(after_opp))
        {
            result.block_live_three_bonus = config.block_live_three_bonus;
            result.reward += result.block_live_three_bonus;
        }

        if (winningTotal(before_opp) > 0 && winningTotal(after_opp) > 0)
        {
            result.unresolved_winning_threat_penalty = -config.unresolved_winning_threat_penalty;
            result.reward += result.unresolved_winning_threat_penalty;
        }
        else if (fourThreatTotal(before_opp) > 0 && fourThreatTotal(after_opp) >= fourThreatTotal(before_opp))
        {
            result.unresolved_four_threat_penalty = -config.unresolved_four_threat_penalty;
            result.reward += result.unresolved_four_threat_penalty;
        }
        else if (
            liveThreeTotal(before_opp) > 0 &&
            liveThreeTotal(after_opp) >= liveThreeTotal(before_opp) &&
            fourThreatTotal(after_self) == 0)
        {
            result.unresolved_live_three_fatal_penalty = -config.unresolved_live_three_fatal_penalty;
            result.reward += result.unresolved_live_three_fatal_penalty;
        }

        result.events = buildEventPairs(self_delta, opp_delta);
        return result;
    }

    Stone RewardEvaluator::playerToStone(int player)
    {
        if (player == 1)
        {
            return Stone::kBlack;
        }
        if (player == -1)
        {
            return Stone::kWhite;
        }
        throw std::invalid_argument("player must be 1 or -1");
    }

    RewardEvaluator::DenseCountArray RewardEvaluator::zeroDenseCounts()
    {
        DenseCountArray counts{};
        counts.fill(0);
        return counts;
    }

    std::size_t RewardEvaluator::trackedStateIndex(StateValueId state_id)
    {
        for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index)
        {
            if (kTrackedStateValueIds[index] == state_id)
            {
                return index;
            }
        }
        throw std::invalid_argument("state id is not tracked");
    }

    std::array<RelativeCellState, kDirectionSideCount> RewardEvaluator::extractSideCellsForPlayer(
        py::array_t<std::int8_t> &board_array,
        int row,
        int col,
        Direction direction,
        int player)
    {
        auto board = board_array.unchecked<2>();
        const DirectionLine line = extract_direction_line(
            board.data(0, 0),
            Vec2i{row, col},
            direction,
            playerToStone(player),
            kBoardSize);
        return line_to_side_cells(line);
    }

    void RewardEvaluator::accumulateDenseDelta(
        DenseCountArray *dense_delta,
        const std::array<std::int8_t, kTrackedStateValueCount> &packed_delta)
    {
        for (std::size_t index = 0; index < kTrackedStateValueCount; ++index)
        {
            dense_delta->at(index) += static_cast<int>(packed_delta[index]);
        }
    }

    void RewardEvaluator::accumulateDenseCounts(DenseCountArray *dense_counts, const StateValueCountVector &counts)
    {
        for (const StateValueCount &state_count : counts)
        {
            dense_counts->at(trackedStateIndex(state_count.state_id)) += state_count.count;
        }
    }

    std::vector<std::pair<int, int>> RewardEvaluator::buildEventPairs(
        const DenseCountArray &self_delta,
        const DenseCountArray &opp_delta)
    {
        std::vector<std::pair<int, int>> events;
        for (std::size_t index = 0; index < kTrackedStateValueCount; ++index)
        {
            if (self_delta[index] != 0)
            {
                events.emplace_back(static_cast<int>(kTrackedStateValueIds[index]), self_delta[index]);
            }
            if (opp_delta[index] != 0)
            {
                events.emplace_back(kOpponentEventOffset + static_cast<int>(kTrackedStateValueIds[index]), opp_delta[index]);
            }
        }
        return events;
    }

    double RewardEvaluator::scoreDelta(const DenseCountArray &delta, const RewardConfig &config, bool invert_sign)
    {
        double score = 0.0;
        for (std::size_t index = 0; index < kTrackedStateValueCount; ++index)
        {
            const int signed_delta = invert_sign ? -delta[index] : delta[index];
            score += static_cast<double>(signed_delta) * config.state_scores[index];
        }
        return score;
    }

    int RewardEvaluator::sumStates(const DenseCountArray &counts, std::initializer_list<StateValueId> state_ids)
    {
        int total = 0;
        for (StateValueId state_id : state_ids)
        {
            total += counts[trackedStateIndex(state_id)];
        }
        return total;
    }

    int RewardEvaluator::liveThreeTotal(const DenseCountArray &counts)
    {
        return sumStates(counts, {
                                     StateValueId::kLiveThree,
                                     StateValueId::kLiveThreeGap1,
                                     StateValueId::kLiveThreeGap2,
                                 });
    }

    int RewardEvaluator::fourThreatTotal(const DenseCountArray &counts)
    {
        return sumStates(counts, {
                                     StateValueId::kLiveFour,
                                     StateValueId::kLiveFourGap1,
                                     StateValueId::kSleepFour,
                                     StateValueId::kSleepFourGap1,
                                 });
    }

    int RewardEvaluator::winningTotal(const DenseCountArray &counts)
    {
        return sumStates(counts, {StateValueId::kFive});
    }

    RewardEvaluator::DenseCountArray RewardEvaluator::applyDelta(
        const DenseCountArray &previous,
        const DenseCountArray &delta)
    {
        DenseCountArray updated = previous;
        for (std::size_t index = 0; index < kTrackedStateValueCount; ++index)
        {
            updated[index] += delta[index];
        }
        return updated;
    }

    py::array_t<std::int8_t> RewardEvaluator::copyBoardAndApplyMove(
        const py::array_t<std::int8_t> &board_before,
        int row,
        int col,
        int player)
    {
        py::array_t<std::int8_t> board_after({kBoardSize, kBoardSize});
        auto before = board_before.unchecked<2>();
        auto after = board_after.mutable_unchecked<2>();
        for (int r = 0; r < kBoardSize; ++r)
        {
            for (int c = 0; c < kBoardSize; ++c)
            {
                after(r, c) = before(r, c);
            }
        }
        after(row, col) = static_cast<std::int8_t>(player);
        return board_after;
    }

    py::list list_state_values()
    {
        py::list payload;
        for (const StateValueDef &def : state_value_definitions())
        {
            py::dict item;
            item["id"] = static_cast<int>(def.state_id);
            item["name"] = py::str(def.name);
            item["description"] = py::str(def.description);
            payload.append(item);
        }
        return payload;
    }

    py::list decode_reward_events(py::iterable events_obj)
    {
        py::list payload;
        for (py::handle item : events_obj)
        {
            const py::tuple event = py::cast<py::tuple>(item);
            if (event.size() != 2)
            {
                throw py::value_error("each reward event must be a pair: (event_id, count)");
            }

            const int raw_event_id = py::cast<int>(event[0]);
            const int count = py::cast<int>(event[1]);
            const bool is_opponent = raw_event_id >= kOpponentEventOffset;
            const int state_id_value = is_opponent ? raw_event_id - kOpponentEventOffset : raw_event_id;
            const StateValueId state_id = static_cast<StateValueId>(state_id_value);

            py::dict decoded;
            decoded["event_id"] = raw_event_id;
            decoded["count"] = count;
            decoded["side"] = is_opponent ? "opponent" : "self";
            decoded["state_id"] = state_id_value;
            decoded["state_name"] = py::str(state_value_name(state_id));
            decoded["state_description"] = py::str(state_value_description(state_id));
            payload.append(decoded);
        }
        return payload;
    }

    py::dict debug_encode_direction_side_states(py::iterable states_obj)
    {
        std::array<RelativeCellState, kDirectionSideCount> states{};
        int index = 0;
        for (py::handle item : states_obj)
        {
            if (index >= kDirectionSideCount)
            {
                throw py::value_error("expected at most 10 side states");
            }
            const int value = py::cast<int>(item);
            if (value < 0 || value > 2)
            {
                throw py::value_error("side state must be 0, 1, or 2");
            }
            states[static_cast<std::size_t>(index++)] = static_cast<RelativeCellState>(value);
        }
        if (index != kDirectionSideCount)
        {
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

    py::dict debug_decode_direction_lookup_key(std::uint32_t lookup_key)
    {
        const EncodedDirectionState decoded = decode_direction_lookup_key(lookup_key);
        py::list states;
        for (RelativeCellState state : decoded.side_cells)
        {
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

    py::list debug_classify_direction_side_states(py::iterable states_obj)
    {
        std::array<RelativeCellState, kDirectionSideCount> states{};
        int index = 0;
        for (py::handle item : states_obj)
        {
            if (index >= kDirectionSideCount)
            {
                throw py::value_error("expected at most 10 side states");
            }
            const int value = py::cast<int>(item);
            if (value < 0 || value > 2)
            {
                throw py::value_error("side state must be 0, 1, or 2");
            }
            states[static_cast<std::size_t>(index++)] = static_cast<RelativeCellState>(value);
        }
        if (index != kDirectionSideCount)
        {
            throw py::value_error("expected exactly 10 side states");
        }

        const StateValueCountVector counts = classify_restored_direction_state_counts(states, kDirectionRadius, true);
        py::list payload;
        for (const StateValueCount &state_count : counts)
        {
            if (state_count.count > 0)
            {
                payload.append(py::str(state_value_name(state_count.state_id)));
            }
        }
        return payload;
    }

    py::dict evaluate_reward(
        py::array_t<std::int8_t> board_before,
        int row,
        int col,
        int player)
    {
        const RewardEvaluator::DenseCountArray before_self = GameStateStore::countBoardStatesForPlayer(board_before, player);
        const RewardEvaluator::DenseCountArray before_opp = GameStateStore::countBoardStatesForPlayer(board_before, -player);
        RewardResult result = RewardEvaluator::instance().evaluate(board_before, row, col, player, before_self, before_opp);

        py::dict payload;
        payload["reward"] = result.reward;
        payload["offense_score"] = result.offense_score;
        payload["defense_score"] = result.defense_score;
        payload["special_rewards"] = build_special_rewards_payload(result);
        payload["events"] = py::cast(result.events);
        return payload;
    }

    py::dict evaluate_env_reward(int env_id, int row, int col, int player)
    {
        GameStateStore &store = GameStateStore::instance();
        py::array_t<std::int8_t> board_before = store.board(env_id);
        const RewardEvaluator::DenseCountArray &before_self = store.countsForPlayer(env_id, player);
        const RewardEvaluator::DenseCountArray &before_opp = store.countsForPlayer(env_id, -player);
        RewardResult result = RewardEvaluator::instance().evaluate(board_before, row, col, player, before_self, before_opp);

        py::dict payload;
        payload["reward"] = result.reward;
        payload["offense_score"] = result.offense_score;
        payload["defense_score"] = result.defense_score;
        payload["special_rewards"] = build_special_rewards_payload(result);
        payload["events"] = py::cast(result.events);
        return payload;
    }

    py::list score_classic_candidates(
        py::array_t<std::int8_t> board_before,
        py::iterable actions_obj,
        int player,
        int thread_batch_size)
    {
        if (thread_batch_size <= 0)
        {
            throw py::value_error("thread_batch_size must be positive");
        }

        std::vector<int> actions;
        for (py::handle item : actions_obj)
        {
            actions.push_back(py::cast<int>(item));
        }

        const FlatBoardArray board = copy_flat_board(board_before);
        std::vector<ClassicCandidateScore> scores(actions.size());
        const int center_row = kBoardSize / 2;
        const int center_col = kBoardSize / 2;

        {
            py::gil_scoped_release release;
            const std::size_t chunk_size = static_cast<std::size_t>(thread_batch_size);
            const std::size_t thread_count = actions.empty() ? 1 : (actions.size() + chunk_size - 1) / chunk_size;
            std::vector<std::thread> threads;
            threads.reserve(thread_count);

            auto worker = [&](std::size_t start, std::size_t end)
            {
                for (std::size_t index = start; index < end; ++index)
                {
                    const int action = actions[index];
                    ClassicCandidateScore candidate{};
                    candidate.action = action;
                    const int row = action / kBoardSize;
                    const int col = action % kBoardSize;
                    if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize)
                    {
                        scores[index] = candidate;
                        continue;
                    }
                    if (board[static_cast<std::size_t>(action)] != kEmpty)
                    {
                        scores[index] = candidate;
                        continue;
                    }

                    const RewardEvaluator::DenseCountArray self_delta = compute_self_delta_for_move(board, row, col, player);
                    const RewardEvaluator::DenseCountArray opp_delta = compute_self_delta_for_move(board, row, col, -player);
                    candidate.self_win = self_delta[tracked_state_index_local(StateValueId::kFive)] > 0;
                    candidate.opp_win = opp_delta[tracked_state_index_local(StateValueId::kFive)] > 0;
                    candidate.score = classic_state_score(self_delta);
                    candidate.score -= std::hypot(static_cast<double>(row - center_row), static_cast<double>(col - center_col));
                    scores[index] = candidate;
                }
            };

            for (std::size_t start = 0; start < actions.size(); start += chunk_size)
            {
                const std::size_t end = std::min(start + chunk_size, actions.size());
                threads.emplace_back(worker, start, end);
            }
            for (std::thread &thread : threads)
            {
                thread.join();
            }
        }

        py::list payload;
        for (const ClassicCandidateScore &score : scores)
        {
            py::dict item;
            item["action"] = score.action;
            item["score"] = score.score;
            item["self_win"] = score.self_win;
            item["opp_win"] = score.opp_win;
            payload.append(item);
        }
        return payload;
    }

    py::list score_reward_candidates(
        py::array_t<std::int8_t> board_before,
        py::iterable actions_obj,
        int player,
        int thread_batch_size)
    {
        if (thread_batch_size <= 0)
        {
            throw py::value_error("thread_batch_size must be positive");
        }

        std::vector<int> actions;
        for (py::handle item : actions_obj)
        {
            actions.push_back(py::cast<int>(item));
        }

        const FlatBoardArray board = copy_flat_board(board_before);
        const RewardEvaluator::DenseCountArray before_self = GameStateStore::countBoardStatesForPlayer(board_before, player);
        const RewardEvaluator::DenseCountArray before_opp = GameStateStore::countBoardStatesForPlayer(board_before, -player);
        std::vector<RewardCandidateScore> scores(actions.size());

        {
            py::gil_scoped_release release;
            const std::size_t chunk_size = static_cast<std::size_t>(thread_batch_size);
            const std::size_t thread_count = actions.empty() ? 1 : (actions.size() + chunk_size - 1) / chunk_size;
            std::vector<std::thread> threads;
            threads.reserve(thread_count);

            auto worker = [&](std::size_t start, std::size_t end)
            {
                for (std::size_t index = start; index < end; ++index)
                {
                    const int action = actions[index];
                    RewardCandidateScore candidate{};
                    candidate.action = action;
                    const int row = action / kBoardSize;
                    const int col = action % kBoardSize;
                    if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize)
                    {
                        scores[index] = candidate;
                        continue;
                    }
                    if (board[static_cast<std::size_t>(action)] != kEmpty)
                    {
                        scores[index] = candidate;
                        continue;
                    }

                    const RewardEvaluator::DenseCountArray self_delta = compute_self_delta_for_move(board, row, col, player);
                    RewardEvaluator::DenseCountArray opp_delta = zero_dense_counts_local();
                    for (Direction direction : kDirections)
                    {
                        const auto opp_side_cells = extract_side_cells_from_flat_board(board, row, col, direction, -player);
                        const PackedDirectionDeltaEntry &opp_entry = lookup_direction_delta_entry(opp_side_cells);
                        accumulate_dense_delta_local(&opp_delta, opp_entry.other_delta);
                    }
                    FlatBoardArray after_board = board;
                    after_board[static_cast<std::size_t>(action)] = static_cast<std::int8_t>(player);
                    const RewardEvaluator::DenseCountArray after_self =
                        GameStateStore::countBoardStatesForPlayer(after_board, player);
                    const RewardEvaluator::DenseCountArray after_opp =
                        GameStateStore::countBoardStatesForPlayer(after_board, -player);
                    const RewardConfig &config = current_reward_config();
                    candidate.offense_score = score_delta_local(self_delta, config, false);
                    candidate.defense_score = score_delta_local(opp_delta, config, true);
                    candidate.reward = config.step_penalty + candidate.offense_score + candidate.defense_score;

                    if (four_threat_total_local(before_self) > 0 && winning_total_local(after_self) == 0)
                    {
                        candidate.reward -= config.missed_immediate_win_penalty;
                    }
                    if (live_three_total_local(self_delta) >= 2)
                    {
                        candidate.reward += config.double_live_three_bonus;
                    }
                    if (winning_total_local(before_opp) > 0 && winning_total_local(after_opp) == 0)
                    {
                        candidate.reward += config.block_winning_bonus;
                    }
                    else if (four_threat_total_local(before_opp) > four_threat_total_local(after_opp))
                    {
                        candidate.reward += config.block_winning_bonus;
                    }
                    else if (live_three_total_local(before_opp) > live_three_total_local(after_opp))
                    {
                        candidate.reward += config.block_live_three_bonus;
                    }

                    if (winning_total_local(before_opp) > 0 && winning_total_local(after_opp) > 0)
                    {
                        candidate.reward -= config.unresolved_winning_threat_penalty;
                    }
                    else if (
                        four_threat_total_local(before_opp) > 0 &&
                        four_threat_total_local(after_opp) >= four_threat_total_local(before_opp))
                    {
                        candidate.reward -= config.unresolved_four_threat_penalty;
                    }
                    else if (
                        live_three_total_local(before_opp) > 0 &&
                        live_three_total_local(after_opp) >= live_three_total_local(before_opp) &&
                        four_threat_total_local(after_self) == 0)
                    {
                        candidate.reward -= config.unresolved_live_three_fatal_penalty;
                    }
                    scores[index] = candidate;
                }
            };

            for (std::size_t start = 0; start < actions.size(); start += chunk_size)
            {
                const std::size_t end = std::min(start + chunk_size, actions.size());
                threads.emplace_back(worker, start, end);
            }
            for (std::thread &thread : threads)
            {
                thread.join();
            }
        }

        py::list payload;
        for (const RewardCandidateScore &score : scores)
        {
            py::dict item;
            item["action"] = score.action;
            item["reward"] = score.reward;
            item["offense_score"] = score.offense_score;
            item["defense_score"] = score.defense_score;
            payload.append(item);
        }
        return payload;
    }

} // namespace gomoku

PYBIND11_MODULE(_cpp_backend, module)
{
    module.doc() = "C++ backend for Gomoku reward evaluation";
    module.def(
        "evaluate_reward",
        &gomoku::evaluate_reward,
        py::arg("board_before"),
        py::arg("row"),
        py::arg("col"),
        py::arg("player"));
    module.def(
        "evaluate_env_reward",
        &gomoku::evaluate_env_reward,
        py::arg("env_id"),
        py::arg("row"),
        py::arg("col"),
        py::arg("player"));
    module.def(
        "score_classic_candidates",
        &gomoku::score_classic_candidates,
        py::arg("board_before"),
        py::arg("actions"),
        py::arg("player"),
        py::arg("thread_batch_size") = 6);
    module.def(
        "score_reward_candidates",
        &gomoku::score_reward_candidates,
        py::arg("board_before"),
        py::arg("actions"),
        py::arg("player"),
        py::arg("thread_batch_size") = 6);
    module.def("reset_env_state", &gomoku::reset_env_state, py::arg("env_id"), py::arg("board"));
    module.def("clear_env_state", &gomoku::clear_env_state, py::arg("env_id"));
    module.def("apply_env_move", &gomoku::apply_env_move, py::arg("env_id"), py::arg("row"), py::arg("col"), py::arg("player"));
    module.def("env_done", &gomoku::env_done, py::arg("env_id"));
    module.def("env_winner", &gomoku::env_winner, py::arg("env_id"));
    module.def("list_state_values", &gomoku::list_state_values);
    module.def("decode_reward_events", &gomoku::decode_reward_events, py::arg("events"));
    module.def("debug_encode_direction_side_states", &gomoku::debug_encode_direction_side_states, py::arg("states"));
    module.def("debug_decode_direction_lookup_key", &gomoku::debug_decode_direction_lookup_key, py::arg("lookup_key"));
    module.def("debug_classify_direction_side_states", &gomoku::debug_classify_direction_side_states, py::arg("states"));
}
