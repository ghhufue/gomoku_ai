#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <string_view>
#include <utility>

namespace gomoku {

enum class StateStoneCount : std::int32_t {
    kOne = 1,
    kTwo = 2,
    kThree = 3,
    kFour = 4,
    kFive = 5,
    kSystem = 15,
};

enum class StatePatternKind : std::int32_t {
    kLive = 1,
    kSleep = 2,
    kDead = 3,
    kTerminal = 15,
};

enum class StateGapKind : std::int32_t {
    kNone = 0,
    kGap1 = 1,
    kGap2 = 2,
    kIllegal = 15,
};

constexpr std::int32_t make_state_value_code(
    StateStoneCount stone_count,
    StatePatternKind pattern_kind,
    StateGapKind gap_kind
) {
    return (static_cast<std::int32_t>(stone_count) << 8)
        | (static_cast<std::int32_t>(pattern_kind) << 4)
        | static_cast<std::int32_t>(gap_kind);
}

enum class StateValueId : std::int32_t {
    kLiveOne = make_state_value_code(StateStoneCount::kOne, StatePatternKind::kLive, StateGapKind::kNone),
    kSleepOne = make_state_value_code(StateStoneCount::kOne, StatePatternKind::kSleep, StateGapKind::kNone),
    kDeadOne = make_state_value_code(StateStoneCount::kOne, StatePatternKind::kDead, StateGapKind::kNone),

    kLiveTwo = make_state_value_code(StateStoneCount::kTwo, StatePatternKind::kLive, StateGapKind::kNone),
    kLiveTwoGap1 = make_state_value_code(StateStoneCount::kTwo, StatePatternKind::kLive, StateGapKind::kGap1),
    kSleepTwo = make_state_value_code(StateStoneCount::kTwo, StatePatternKind::kSleep, StateGapKind::kNone),
    kSleepTwoGap1 = make_state_value_code(StateStoneCount::kTwo, StatePatternKind::kSleep, StateGapKind::kGap1),
    kDeadTwo = make_state_value_code(StateStoneCount::kTwo, StatePatternKind::kDead, StateGapKind::kNone),
    kDeadTwoGap1 = make_state_value_code(StateStoneCount::kTwo, StatePatternKind::kDead, StateGapKind::kGap1),

    kLiveThree = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kLive, StateGapKind::kNone),
    kLiveThreeGap1 = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kLive, StateGapKind::kGap1),
    kLiveThreeGap2 = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kLive, StateGapKind::kGap2),
    kSleepThree = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kSleep, StateGapKind::kNone),
    kSleepThreeGap1 = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kSleep, StateGapKind::kGap1),
    kSleepThreeGap2 = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kSleep, StateGapKind::kGap2),
    kDeadThree = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kDead, StateGapKind::kNone),
    kDeadThreeGap1 = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kDead, StateGapKind::kGap1),
    kDeadThreeGap2 = make_state_value_code(StateStoneCount::kThree, StatePatternKind::kDead, StateGapKind::kGap2),

    kLiveFour = make_state_value_code(StateStoneCount::kFour, StatePatternKind::kLive, StateGapKind::kNone),
    kLiveFourGap1 = make_state_value_code(StateStoneCount::kFour, StatePatternKind::kLive, StateGapKind::kGap1),
    kSleepFour = make_state_value_code(StateStoneCount::kFour, StatePatternKind::kSleep, StateGapKind::kNone),
    kSleepFourGap1 = make_state_value_code(StateStoneCount::kFour, StatePatternKind::kSleep, StateGapKind::kGap1),
    kDeadFour = make_state_value_code(StateStoneCount::kFour, StatePatternKind::kDead, StateGapKind::kNone),
    kDeadFourGap1 = make_state_value_code(StateStoneCount::kFour, StatePatternKind::kDead, StateGapKind::kGap1),

    kFive = make_state_value_code(StateStoneCount::kFive, StatePatternKind::kTerminal, StateGapKind::kNone),
    kIllegal = make_state_value_code(StateStoneCount::kSystem, StatePatternKind::kTerminal, StateGapKind::kIllegal),
};

struct StateValueCount {
    StateValueId state_id;
    int count;
};

struct StateValueDelta {
    StateValueId state_id;
    int delta;
};

struct StateValueDef {
    StateValueId state_id;
    std::string_view name;
    std::string_view description;
};

using StateValuePair = std::pair<int, int>;

constexpr std::array<StateValueId, 25> kTrackedStateValueIds = {
    StateValueId::kLiveOne,
    StateValueId::kSleepOne,
    StateValueId::kDeadOne,
    StateValueId::kLiveTwo,
    StateValueId::kLiveTwoGap1,
    StateValueId::kSleepTwo,
    StateValueId::kSleepTwoGap1,
    StateValueId::kDeadTwo,
    StateValueId::kDeadTwoGap1,
    StateValueId::kLiveThree,
    StateValueId::kLiveThreeGap1,
    StateValueId::kLiveThreeGap2,
    StateValueId::kSleepThree,
    StateValueId::kSleepThreeGap1,
    StateValueId::kSleepThreeGap2,
    StateValueId::kDeadThree,
    StateValueId::kDeadThreeGap1,
    StateValueId::kDeadThreeGap2,
    StateValueId::kLiveFour,
    StateValueId::kLiveFourGap1,
    StateValueId::kSleepFour,
    StateValueId::kSleepFourGap1,
    StateValueId::kDeadFour,
    StateValueId::kDeadFourGap1,
    StateValueId::kFive,
};

constexpr std::size_t kTrackedStateValueCount = kTrackedStateValueIds.size();

using StateValueCountVector = std::array<StateValueCount, kTrackedStateValueCount>;
using StateValueDeltaVector = std::array<StateValueDelta, kTrackedStateValueCount>;

class StateValueRegistry {
public:
    const std::array<StateValueDef, 24>& definitions() const;
    std::string_view name(StateValueId state_id) const;
    std::string_view description(StateValueId state_id) const;

    static const StateValueRegistry& instance();
};

std::string_view state_value_name(StateValueId state_id);
std::string_view state_value_description(StateValueId state_id);

const std::array<StateValueDef, 24>& state_value_definitions();

StateValuePair to_state_value_pair(StateValueCount state);

}  // namespace gomoku
