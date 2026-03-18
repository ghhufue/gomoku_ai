#pragma once

#include "StateValueRegistry.h"

#include <array>
#include <string>

namespace gomoku {

struct RewardConfig {
    std::array<double, kTrackedStateValueCount> state_scores{};

    double step_penalty = -1.0;
    double double_live_three_bonus = 10.0;
    double block_winning_bonus = 100.0;
    double block_live_four_bonus = 30.0;
    double block_live_three_bonus = 20.0;
    double unresolved_winning_threat_penalty = 120.0;
    double unresolved_four_threat_penalty = 45.0;
    double unresolved_live_three_fatal_penalty = 200.0;
    double missed_immediate_win_penalty = 150.0;
};

class RewardConfigStore {
public:
    const RewardConfig& current();
    void loadFromFile(const std::string& path);
    std::string defaultPath() const;
    bool isLoaded() const;

    static RewardConfigStore& instance();

private:
    RewardConfig config_{};
    bool loaded_ = false;
};

const RewardConfig& current_reward_config();

void load_reward_config_from_file(const std::string& path);

std::string default_reward_config_path();

}  // namespace gomoku
