#include <algorithm>
#include <array>
#include <cstdint>
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

    return pattern;
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
        "immediate_winning_actions",
        &immediate_winning_actions,
        py::arg("board"),
        py::arg("player"),
        py::arg("actions") = py::none()
    );
    module.def("threat_summary", &threat_summary, py::arg("board"), py::arg("player"), py::arg("actions") = py::none());
    module.def("affected_actions", &affected_actions, py::arg("board"), py::arg("row"), py::arg("col"), py::arg("radius") = 5);
}
