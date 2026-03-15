#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr int BOARD_SIZE = 15;
constexpr int EMPTY = 0;
constexpr std::array<std::array<int, 2>, 4> DIRS = {{{1, 0}, {0, 1}, {1, 1}, {1, -1}}};
constexpr int ANALYZE_RADIUS = 5;
constexpr int ANALYZE_CENTER = ANALYZE_RADIUS;

struct Pattern {
    int five = 0;
    int live_four = 0;
    int rush_four = 0;
    int live_three = 0;
    int sleep_three = 0;
    int live_two = 0;
};

struct ShapeScaleConfig {
    double live_two_contiguous_scale = 1.0;
    double live_two_gap1_scale = 0.5;
    double live_two_gap2_scale = 0.15;
    double live_three_contiguous_scale = 1.0;
    double live_three_gap1_scale = 0.7;
    double live_three_gap2_scale = 0.3;
    double sleep_three_contiguous_scale = 1.0;
    double sleep_three_gap1_scale = 0.7;
    double sleep_three_gap2_scale = 0.3;
    double live_four_contiguous_scale = 1.0;
    double live_four_gap1_scale = 0.5;
    double rush_four_contiguous_scale = 1.0;
    double rush_four_gap1_scale = 0.5;
};

struct ShapeDetail {
    double scale = 0.0;
    const char* bucket = nullptr;
    std::string window{};
};

struct DirectionShapeDetails {
    ShapeDetail live_two{};
    ShapeDetail live_three{};
    ShapeDetail sleep_three{};
    ShapeDetail live_four{};
    ShapeDetail rush_four{};
};

constexpr std::array<std::string_view, 4> LIVE_FOUR_PATTERNS = {
    "_XXXX_",
    "_XXX_X_",
    "_XX_XX_",
    "_X_XXX_",
};

constexpr std::array<std::string_view, 3> LIVE_THREE_PATTERNS = {
    "_XXX_",
    "_XX_X_",
    "_X_XX_",
};

constexpr std::array<std::string_view, 5> DETAILED_LIVE_THREE_PATTERNS = {
    "_XXX_",
    "_XX_X_",
    "_X_XX_",
    "_XX__X_",
    "_X__XX_",
};

constexpr std::array<std::string_view, 8> SLEEP_THREE_PATTERNS = {
    "OXXX__",
    "__XXXO",
    "O_XXX_",
    "_XXX_O",
    "OX_XX_",
    "_XX_XO",
    "OXX_X_",
    "_X_XXO",
};

constexpr std::array<std::string_view, 16> DETAILED_SLEEP_THREE_PATTERNS = {
    "OXXX__",
    "__XXXO",
    "O_XXX_",
    "_XXX_O",
    "OX_XX_",
    "_XX_XO",
    "OXX_X_",
    "_X_XXO",
    "OXX__X",
    "X__XXO",
    "OX__XX",
    "XX__XO",
    "OXX__X_",
    "_X__XXO",
    "OX__XX_",
    "_XX__XO",
};

constexpr std::array<std::string_view, 5> LIVE_TWO_PATTERNS = {
    "_XX_",
    "_X_X_",
    "__XX__",
    "__X_X__",
    "_X__X_",
};

template <std::size_t N>
bool contains_pattern(std::string_view window, const std::array<std::string_view, N>& patterns) {
    for (std::string_view pattern : patterns) {
        if (window == pattern) {
            return true;
        }
    }
    return false;
}

int gap_bucket_id(std::string_view window) {
    int first = -1;
    int last = -1;
    int stones = 0;
    for (int index = 0; index < static_cast<int>(window.size()); ++index) {
        if (window[index] == 'X') {
            if (first < 0) {
                first = index;
            }
            last = index;
            ++stones;
        }
    }
    if (stones <= 1 || first < 0 || last < 0) {
        return 0;
    }
    const int internal_empties = (last - first + 1) - stones;
    if (internal_empties <= 0) {
        return 0;
    }
    if (internal_empties == 1) {
        return 1;
    }
    return 2;
}

const char* bucket_name(int bucket) {
    switch (bucket) {
        case 0:
            return "contiguous";
        case 1:
            return "gap1";
        default:
            return "gap2";
    }
}

double shape_scale(const ShapeScaleConfig& config, const std::string& family, int bucket) {
    if (family == "live_two") {
        return bucket == 0 ? config.live_two_contiguous_scale : (bucket == 1 ? config.live_two_gap1_scale : config.live_two_gap2_scale);
    }
    if (family == "live_three") {
        return bucket == 0 ? config.live_three_contiguous_scale : (bucket == 1 ? config.live_three_gap1_scale : config.live_three_gap2_scale);
    }
    if (family == "sleep_three") {
        return bucket == 0 ? config.sleep_three_contiguous_scale : (bucket == 1 ? config.sleep_three_gap1_scale : config.sleep_three_gap2_scale);
    }
    if (family == "live_four") {
        return bucket == 0 ? config.live_four_contiguous_scale : config.live_four_gap1_scale;
    }
    return bucket == 0 ? config.rush_four_contiguous_scale : config.rush_four_gap1_scale;
}

ShapeScaleConfig parse_scale_config(const py::dict& scales) {
    ShapeScaleConfig config{};
    auto get_value = [&](const char* key, double current) {
        if (scales.contains(key)) {
            return py::cast<double>(scales[key]);
        }
        return current;
    };
    config.live_two_contiguous_scale = get_value("live_two_contiguous_scale", config.live_two_contiguous_scale);
    config.live_two_gap1_scale = get_value("live_two_gap1_scale", config.live_two_gap1_scale);
    config.live_two_gap2_scale = get_value("live_two_gap2_scale", config.live_two_gap2_scale);
    config.live_three_contiguous_scale = get_value("live_three_contiguous_scale", config.live_three_contiguous_scale);
    config.live_three_gap1_scale = get_value("live_three_gap1_scale", config.live_three_gap1_scale);
    config.live_three_gap2_scale = get_value("live_three_gap2_scale", config.live_three_gap2_scale);
    config.sleep_three_contiguous_scale = get_value("sleep_three_contiguous_scale", config.sleep_three_contiguous_scale);
    config.sleep_three_gap1_scale = get_value("sleep_three_gap1_scale", config.sleep_three_gap1_scale);
    config.sleep_three_gap2_scale = get_value("sleep_three_gap2_scale", config.sleep_three_gap2_scale);
    config.live_four_contiguous_scale = get_value("live_four_contiguous_scale", config.live_four_contiguous_scale);
    config.live_four_gap1_scale = get_value("live_four_gap1_scale", config.live_four_gap1_scale);
    config.rush_four_contiguous_scale = get_value("rush_four_contiguous_scale", config.rush_four_contiguous_scale);
    config.rush_four_gap1_scale = get_value("rush_four_gap1_scale", config.rush_four_gap1_scale);
    return config;
}

template <typename Board>
char board_symbol(const Board& board, int row, int col, int player) {
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
        return 'O';
    }
    const int value = board(row, col);
    if (value == player) {
        return 'X';
    }
    if (value == EMPTY) {
        return '_';
    }
    return 'O';
}

template <typename Board>
std::array<char, 2 * ANALYZE_RADIUS + 1> direction_line(
    const Board& board, int row, int col, int dr, int dc, int player
) {
    std::array<char, 2 * ANALYZE_RADIUS + 1> line{};
    for (int offset = -ANALYZE_RADIUS; offset <= ANALYZE_RADIUS; ++offset) {
        const int index = offset + ANALYZE_RADIUS;
        line[index] = board_symbol(board, row + offset * dr, col + offset * dc, player);
    }
    return line;
}

template <typename Board>
Pattern analyze_direction(const Board& board, int row, int col, int dr, int dc, int player) {
    const auto line = direction_line(board, row, col, dr, dc, player);

    int contiguous = 1;
    for (int left = ANALYZE_CENTER - 1; left >= 0 && line[left] == 'X'; --left) {
        ++contiguous;
    }
    for (int right = ANALYZE_CENTER + 1; right < static_cast<int>(line.size()) && line[right] == 'X'; ++right) {
        ++contiguous;
    }
    if (contiguous >= 5) {
        return Pattern{1, 0, 0, 0, 0, 0};
    }

    Pattern pattern{};
    for (int length : {4, 5, 6, 7}) {
        const int start_min = std::max(0, ANALYZE_CENTER - length + 1);
        const int start_max = std::min(ANALYZE_CENTER, static_cast<int>(line.size()) - length);
        for (int start = start_min; start <= start_max; ++start) {
            std::array<char, 7> window_buffer{};
            int stones = 0;
            int empties = 0;
            for (int offset = 0; offset < length; ++offset) {
                const char symbol = line[start + offset];
                window_buffer[offset] = symbol;
                stones += (symbol == 'X');
                empties += (symbol == '_');
            }
            const std::string_view window(window_buffer.data(), length);

            if (contains_pattern(window, LIVE_FOUR_PATTERNS)) {
                pattern.live_four = 1;
                continue;
            }
            if (stones == 4 && empties == 1) {
                pattern.rush_four = 1;
                continue;
            }
            if (contains_pattern(window, LIVE_THREE_PATTERNS)) {
                pattern.live_three = 1;
                continue;
            }
            if (contains_pattern(window, SLEEP_THREE_PATTERNS)) {
                pattern.sleep_three = 1;
                continue;
            }
            if (contains_pattern(window, LIVE_TWO_PATTERNS)) {
                pattern.live_two = 1;
            }
        }
    }

    if (pattern.live_four) {
        pattern.rush_four = 0;
    }

    return pattern;
}

template <typename Board>
DirectionShapeDetails analyze_direction_shape_details(
    const Board& board, int row, int col, int dr, int dc, int player, const ShapeScaleConfig& config
) {
    const auto line = direction_line(board, row, col, dr, dc, player);
    DirectionShapeDetails details{};

    auto update_detail = [&](ShapeDetail& detail, const std::string& family, std::string_view window) {
        const int bucket = gap_bucket_id(window);
        const double scale = shape_scale(config, family, bucket);
        if (scale > detail.scale) {
            detail.scale = scale;
            detail.bucket = bucket_name(bucket);
            detail.window = std::string(window);
        }
    };

    for (int length : {4, 5, 6, 7}) {
        const int start_min = std::max(0, ANALYZE_CENTER - length + 1);
        const int start_max = std::min(ANALYZE_CENTER, static_cast<int>(line.size()) - length);
        for (int start = start_min; start <= start_max; ++start) {
            std::array<char, 7> window_buffer{};
            int stones = 0;
            int empties = 0;
            for (int offset = 0; offset < length; ++offset) {
                const char symbol = line[start + offset];
                window_buffer[offset] = symbol;
                stones += (symbol == 'X');
                empties += (symbol == '_');
            }
            const std::string_view window(window_buffer.data(), length);

            if (contains_pattern(window, LIVE_FOUR_PATTERNS)) {
                update_detail(details.live_four, "live_four", window);
                continue;
            }
            if (stones == 4 && empties == 1) {
                update_detail(details.rush_four, "rush_four", window);
                continue;
            }
            if (contains_pattern(window, DETAILED_LIVE_THREE_PATTERNS)) {
                update_detail(details.live_three, "live_three", window);
                continue;
            }
            if (contains_pattern(window, DETAILED_SLEEP_THREE_PATTERNS)) {
                update_detail(details.sleep_three, "sleep_three", window);
                continue;
            }
            if (contains_pattern(window, LIVE_TWO_PATTERNS)) {
                update_detail(details.live_two, "live_two", window);
            }
        }
    }

    if (details.live_four.scale > 0.0) {
        details.rush_four = ShapeDetail{};
    }
    return details;
}

template <typename Board>
Pattern classify_move_counts_impl(const Board& board, int row, int col, int player) {
    Pattern total{};
    for (const auto& dir : DIRS) {
        const Pattern pattern = analyze_direction(board, row, col, dir[0], dir[1], player);
        total.five |= pattern.five;
        total.live_four += pattern.live_four;
        total.rush_four += pattern.rush_four;
        total.live_three += pattern.live_three;
        total.sleep_three += pattern.sleep_three;
        total.live_two += pattern.live_two;
    }
    return total;
}

py::tuple pattern_to_tuple(const Pattern& pattern) {
    return py::make_tuple(
        pattern.five,
        pattern.live_four,
        pattern.rush_four,
        pattern.live_three,
        pattern.sleep_three,
        pattern.live_two
    );
}

py::dict shape_detail_to_dict(const ShapeDetail& detail) {
    py::dict payload;
    payload["scale"] = detail.scale;
    payload["bucket"] = detail.bucket ? py::str(detail.bucket) : py::str("");
    payload["window"] = detail.window.empty() ? py::str("") : py::str(detail.window);
    return payload;
}

std::vector<int> collect_actions(py::array_t<std::int8_t>& board_array, py::object actions_obj) {
    auto board = board_array.unchecked<2>();
    std::vector<int> actions;
    if (actions_obj.is_none()) {
        actions.reserve(BOARD_SIZE * BOARD_SIZE);
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                if (board(row, col) == EMPTY) {
                    actions.push_back(row * BOARD_SIZE + col);
                }
            }
        }
        return actions;
    }

    for (py::handle item : actions_obj) {
        actions.push_back(py::cast<int>(item));
    }
    return actions;
}

}  // namespace

py::tuple classify_move_counts(py::array_t<std::int8_t> board_array, int row, int col, int player) {
    auto board = board_array.unchecked<2>();
    return pattern_to_tuple(classify_move_counts_impl(board, row, col, player));
}

py::dict classify_move_shape_details(
    py::array_t<std::int8_t> board_array, int row, int col, int player, py::dict scales
) {
    const ShapeScaleConfig config = parse_scale_config(scales);
    auto board = board_array.unchecked<2>();

    std::array<DirectionShapeDetails, 4> directions{};
    std::array<double, 5> family_totals = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<int, 5> active_directions = {0, 0, 0, 0, 0};

    for (int index = 0; index < 4; ++index) {
        const auto& dir = DIRS[index];
        directions[index] = analyze_direction_shape_details(board, row, col, dir[0], dir[1], player, config);
        const std::array<const ShapeDetail*, 5> family_refs = {
            &directions[index].live_two,
            &directions[index].live_three,
            &directions[index].sleep_three,
            &directions[index].live_four,
            &directions[index].rush_four,
        };
        for (int family_index = 0; family_index < 5; ++family_index) {
            family_totals[family_index] += family_refs[family_index]->scale;
            if (family_refs[family_index]->scale > 0.0) {
                active_directions[family_index] += 1;
            }
        }
    }

    py::dict family_totals_dict;
    family_totals_dict["live_two"] = family_totals[0];
    family_totals_dict["live_three"] = family_totals[1];
    family_totals_dict["sleep_three"] = family_totals[2];
    family_totals_dict["live_four"] = family_totals[3];
    family_totals_dict["rush_four"] = family_totals[4];

    py::dict active_directions_dict;
    active_directions_dict["live_two"] = active_directions[0];
    active_directions_dict["live_three"] = active_directions[1];
    active_directions_dict["sleep_three"] = active_directions[2];
    active_directions_dict["live_four"] = active_directions[3];
    active_directions_dict["rush_four"] = active_directions[4];

    py::list directions_list;
    for (int index = 0; index < 4; ++index) {
        py::dict entry;
        entry["direction"] = py::make_tuple(DIRS[index][0], DIRS[index][1]);
        entry["live_two"] = shape_detail_to_dict(directions[index].live_two);
        entry["live_three"] = shape_detail_to_dict(directions[index].live_three);
        entry["sleep_three"] = shape_detail_to_dict(directions[index].sleep_three);
        entry["live_four"] = shape_detail_to_dict(directions[index].live_four);
        entry["rush_four"] = shape_detail_to_dict(directions[index].rush_four);
        directions_list.append(entry);
    }

    py::dict result;
    result["family_totals"] = family_totals_dict;
    result["active_directions"] = active_directions_dict;
    result["directions"] = directions_list;
    return result;
}

py::list immediate_winning_actions(py::array_t<std::int8_t> board_array, int player, py::object actions_obj = py::none()) {
    auto board = board_array.mutable_unchecked<2>();
    const std::vector<int> actions = collect_actions(board_array, actions_obj);
    py::list wins;
    for (int action : actions) {
        const int row = action / BOARD_SIZE;
        const int col = action % BOARD_SIZE;
        if (board(row, col) != EMPTY) {
            continue;
        }
        board(row, col) = static_cast<std::int8_t>(player);
        const Pattern pattern = classify_move_counts_impl(board, row, col, player);
        board(row, col) = EMPTY;
        if (pattern.five) {
            wins.append(action);
        }
    }
    return wins;
}

py::dict threat_summary(py::array_t<std::int8_t> board_array, int player, py::object actions_obj = py::none()) {
    auto board = board_array.mutable_unchecked<2>();
    const std::vector<int> actions = collect_actions(board_array, actions_obj);
    int winning_actions = 0;
    int live_four = 0;
    int rush_four = 0;
    int live_three = 0;
    int sleep_three = 0;
    int live_two = 0;

    for (int action : actions) {
        const int row = action / BOARD_SIZE;
        const int col = action % BOARD_SIZE;
        if (board(row, col) != EMPTY) {
            continue;
        }
        board(row, col) = static_cast<std::int8_t>(player);
        const Pattern pattern = classify_move_counts_impl(board, row, col, player);
        board(row, col) = EMPTY;

        winning_actions += pattern.five;
        live_four += pattern.live_four;
        rush_four += pattern.rush_four;
        live_three += pattern.live_three;
        sleep_three += pattern.sleep_three;
        live_two += pattern.live_two;
    }

    py::dict summary;
    summary["winning_actions"] = winning_actions;
    summary["live_four"] = live_four;
    summary["rush_four"] = rush_four;
    summary["live_three"] = live_three;
    summary["sleep_three"] = sleep_three;
    summary["live_two"] = live_two;
    return summary;
}

py::list affected_actions(py::array_t<std::int8_t> board_array, int row, int col, int radius = 5) {
    auto board = board_array.unchecked<2>();
    py::list actions;
    for (int action_row = 0; action_row < BOARD_SIZE; ++action_row) {
        for (int action_col = 0; action_col < BOARD_SIZE; ++action_col) {
            if (board(action_row, action_col) != EMPTY) {
                continue;
            }
            const int row_delta = action_row - row;
            const int col_delta = action_col - col;
            if (
                (action_row == row && std::abs(col_delta) <= radius) ||
                (action_col == col && std::abs(row_delta) <= radius) ||
                (std::abs(row_delta) == std::abs(col_delta) && std::abs(row_delta) <= radius)
            ) {
                actions.append(action_row * BOARD_SIZE + action_col);
            }
        }
    }
    return actions;
}

PYBIND11_MODULE(_cpp_backend, module) {
    module.doc() = "C++ backend for Gomoku board analysis";
    module.def("classify_move_counts", &classify_move_counts, py::arg("board"), py::arg("row"), py::arg("col"), py::arg("player"));
    module.def(
        "classify_move_shape_details",
        &classify_move_shape_details,
        py::arg("board"),
        py::arg("row"),
        py::arg("col"),
        py::arg("player"),
        py::arg("scales")
    );
    module.def(
        "immediate_winning_actions",
        &immediate_winning_actions,
        py::arg("board"),
        py::arg("player"),
        py::arg("actions") = py::none()
    );
    module.def("threat_summary", &threat_summary, py::arg("board"), py::arg("player"), py::arg("actions") = py::none());
    module.def("affected_actions", &affected_actions, py::arg("board"), py::arg("row"), py::arg("col"), py::arg("radius") = 5);
}
