#include "StateValueRegistry.h"

namespace gomoku {

namespace {

constexpr std::array<StateValueDef, 24> kStateValueDefs = {{
    {StateValueId::kLiveOne, "live_one", "One same-color stone with two open ends."},
    {StateValueId::kSleepOne, "sleep_one", "One same-color stone blocked on one side."},
    {StateValueId::kDeadOne, "dead_one", "One same-color stone blocked on both sides."},

    {StateValueId::kLiveTwo, "live_two", "Two same-color stones with two open ends."},
    {StateValueId::kLiveTwoGap1, "live_two_gap1", "Two same-color stones with one internal gap and two open ends."},
    {StateValueId::kSleepTwo, "sleep_two", "Two same-color stones blocked on one side."},
    {StateValueId::kSleepTwoGap1, "sleep_two_gap1", "Two same-color stones with one internal gap and one blocked side."},
    {StateValueId::kDeadTwo, "dead_two", "Two same-color stones blocked on both sides."},
    {StateValueId::kDeadTwoGap1, "dead_two_gap1", "Two same-color stones with one internal gap and both sides blocked."},

    {StateValueId::kLiveThree, "live_three", "Three same-color stones with two open ends."},
    {StateValueId::kLiveThreeGap1, "live_three_gap1", "Three same-color stones with one internal gap and two open ends."},
    {StateValueId::kLiveThreeGap2, "live_three_gap2", "Three same-color stones with split shape and two open ends."},
    {StateValueId::kSleepThree, "sleep_three", "Three same-color stones blocked on one side."},
    {StateValueId::kSleepThreeGap1, "sleep_three_gap1", "Three same-color stones with one internal gap and one blocked side."},
    {StateValueId::kSleepThreeGap2, "sleep_three_gap2", "Three same-color stones with split shape and one blocked side."},
    {StateValueId::kDeadThree, "dead_three", "Three same-color stones blocked on both sides."},
    {StateValueId::kDeadThreeGap1, "dead_three_gap1", "Three same-color stones with one internal gap and both sides blocked."},
    {StateValueId::kDeadThreeGap2, "dead_three_gap2", "Three same-color stones with split shape and both sides blocked."},

    {StateValueId::kLiveFour, "live_four", "Four same-color stones with two open ends."},
    {StateValueId::kLiveFourGap1, "live_four_gap1", "Four same-color stones with one internal gap and two open ends."},
    {StateValueId::kSleepFour, "sleep_four", "Four same-color stones blocked on one side."},
    {StateValueId::kSleepFourGap1, "sleep_four_gap1", "Four same-color stones with one internal gap and one blocked side."},
    {StateValueId::kDeadFour, "dead_four", "Four same-color stones blocked on both sides."},
    {StateValueId::kDeadFourGap1, "dead_four_gap1", "Four same-color stones with one internal gap and both sides blocked."},

    // Keep five/illegal outside the exported registry for now.
}};

}  // namespace

const StateValueRegistry& StateValueRegistry::instance() {
    static const StateValueRegistry registry{};
    return registry;
}

const std::array<StateValueDef, 24>& StateValueRegistry::definitions() const {
    return kStateValueDefs;
}

std::string_view StateValueRegistry::name(StateValueId state_id) const {
    if (state_id == StateValueId::kFive) {
        return "five";
    }
    if (state_id == StateValueId::kIllegal) {
        return "illegal";
    }
    for (const StateValueDef& def : kStateValueDefs) {
        if (def.state_id == state_id) {
            return def.name;
        }
    }
    return "unknown";
}

std::string_view StateValueRegistry::description(StateValueId state_id) const {
    if (state_id == StateValueId::kFive) {
        return "Five same-color stones in one line.";
    }
    if (state_id == StateValueId::kIllegal) {
        return "Illegal state or move.";
    }
    for (const StateValueDef& def : kStateValueDefs) {
        if (def.state_id == state_id) {
            return def.description;
        }
    }
    return "Unknown state value.";
}

const std::array<StateValueDef, 24>& state_value_definitions() {
    return StateValueRegistry::instance().definitions();
}

std::string_view state_value_name(StateValueId state_id) {
    return StateValueRegistry::instance().name(state_id);
}

std::string_view state_value_description(StateValueId state_id) {
    return StateValueRegistry::instance().description(state_id);
}

StateValuePair to_state_value_pair(StateValueCount state) {
    return {static_cast<int>(state.state_id), state.count};
}

}  // namespace gomoku
