#include "RewardConfigStore.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace gomoku {

namespace {
RewardConfig make_default_reward_config() {
    RewardConfig config{};
    config.state_scores = {
        0.5,   // live_one
        0.2,   // sleep_one
        0.0,   // dead_one
        5.0,   // live_two
        2.5,   // live_two_gap1
        2.0,   // sleep_two
        1.0,   // sleep_two_gap1
        0.0,   // dead_two
        0.0,   // dead_two_gap1
        20.0,  // live_three
        14.0,  // live_three_gap1
        6.0,   // live_three_gap2
        8.0,   // sleep_three
        5.6,   // sleep_three_gap1
        2.4,   // sleep_three_gap2
        0.0,   // dead_three
        0.0,   // dead_three_gap1
        0.0,   // dead_three_gap2
        100.0, // live_four
        50.0,  // live_four_gap1
        50.0,  // sleep_four
        25.0,  // sleep_four_gap1
        0.0,   // dead_four
        0.0,   // dead_four_gap1
        1000.0 // five
    };
    return config;
}

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::unordered_map<std::string, std::size_t> build_state_index_map() {
    std::unordered_map<std::string, std::size_t> indices;
    for (std::size_t index = 0; index < kTrackedStateValueIds.size(); ++index) {
        indices.emplace(std::string(state_value_name(kTrackedStateValueIds[index])), index);
    }
    return indices;
}

const std::unordered_map<std::string, std::size_t>& state_index_map() {
    static const std::unordered_map<std::string, std::size_t> map = build_state_index_map();
    return map;
}

void parse_assignment(const std::string& section, const std::string& line, RewardConfig* config) {
    const std::size_t equal_pos = line.find('=');
    if (equal_pos == std::string::npos) {
        return;
    }

    const std::string key = trim(line.substr(0, equal_pos));
    const std::string value_text = trim(line.substr(equal_pos + 1));
    if (key.empty() || value_text.empty()) {
        return;
    }

    const double value = std::stod(value_text);
    if (section == "reward.state_scores") {
        const auto it = state_index_map().find(key);
        if (it == state_index_map().end()) {
            throw std::runtime_error("unknown state score key in reward config: " + key);
        }
        config->state_scores[it->second] = value;
        return;
    }

    if (section != "reward.special") {
        return;
    }

    if (key == "step_penalty") {
        config->step_penalty = value;
    } else if (key == "double_live_three_bonus") {
        config->double_live_three_bonus = value;
    } else if (key == "block_winning_bonus") {
        config->block_winning_bonus = value;
    } else if (key == "block_live_three_bonus") {
        config->block_live_three_bonus = value;
    } else if (key == "unresolved_winning_threat_penalty") {
        config->unresolved_winning_threat_penalty = value;
    } else if (key == "unresolved_four_threat_penalty") {
        config->unresolved_four_threat_penalty = value;
    } else if (key == "unresolved_live_three_fatal_penalty") {
        config->unresolved_live_three_fatal_penalty = value;
    } else if (key == "missed_immediate_win_penalty") {
        config->missed_immediate_win_penalty = value;
    }
}

}  // namespace

RewardConfigStore& RewardConfigStore::instance() {
    static RewardConfigStore store = [] {
        RewardConfigStore value{};
        value.config_ = make_default_reward_config();
        return value;
    }();
    return store;
}

const RewardConfig& RewardConfigStore::current() {
    if (!loaded_) {
        loadFromFile(defaultPath());
    }
    return config_;
}

void RewardConfigStore::loadFromFile(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open reward config: " + path);
    }

    RewardConfig config = config_;
    std::string section;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        parse_assignment(section, line, &config);
    }

    config_ = config;
    loaded_ = true;
}

std::string RewardConfigStore::defaultPath() const {
    namespace fs = std::filesystem;
    const fs::path cwd_path = fs::current_path() / "configs" / "reward.toml";
    if (fs::exists(cwd_path)) {
        return cwd_path.string();
    }
    return "d:/code/gomoku_ai/configs/reward.toml";
}

bool RewardConfigStore::isLoaded() const {
    return loaded_;
}

std::string default_reward_config_path() {
    return RewardConfigStore::instance().defaultPath();
}

void load_reward_config_from_file(const std::string& path) {
    RewardConfigStore::instance().loadFromFile(path);
}

const RewardConfig& current_reward_config() {
    return RewardConfigStore::instance().current();
}
}  // namespace gomoku
