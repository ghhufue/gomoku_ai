#include "../direction_encoding.h"
#include "../StateValueRegistry.h"
#include "../utils/direction_pattern_lookup.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gomoku::test {

namespace {

struct JsonValue {
    enum class Type {
        kObject,
        kArray,
        kString,
        kNumber,
    };

    Type type = Type::kObject;
    std::map<std::string, JsonValue> object_value{};
    std::vector<JsonValue> array_value{};
    std::string string_value{};
    int number_value = 0;
};

class JsonParser {
public:
    explicit JsonParser(std::string text)
        : text_(std::move(text)) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value();
        skip_whitespace();
        if (position_ != text_.size()) {
            throw std::runtime_error("unexpected trailing json content");
        }
        return value;
    }

private:
    JsonValue parse_value() {
        skip_whitespace();
        if (position_ >= text_.size()) {
            throw std::runtime_error("unexpected end of json");
        }
        const char ch = text_[position_];
        if (ch == '{') {
            return parse_object();
        }
        if (ch == '[') {
            return parse_array();
        }
        if (ch == '"') {
            JsonValue value;
            value.type = JsonValue::Type::kString;
            value.string_value = parse_string();
            return value;
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            JsonValue value;
            value.type = JsonValue::Type::kNumber;
            value.number_value = parse_number();
            return value;
        }
        throw std::runtime_error("unsupported json token");
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::kObject;
        skip_whitespace();
        if (peek('}')) {
            expect('}');
            return value;
        }
        while (true) {
            const std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            value.object_value.emplace(key, parse_value());
            skip_whitespace();
            if (peek('}')) {
                expect('}');
                break;
            }
            expect(',');
            skip_whitespace();
        }
        return value;
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::kArray;
        skip_whitespace();
        if (peek(']')) {
            expect(']');
            return value;
        }
        while (true) {
            value.array_value.push_back(parse_value());
            skip_whitespace();
            if (peek(']')) {
                expect(']');
                break;
            }
            expect(',');
            skip_whitespace();
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (position_ < text_.size()) {
            const char ch = text_[position_++];
            if (ch == '"') {
                return result;
            }
            if (ch == '\\') {
                if (position_ >= text_.size()) {
                    throw std::runtime_error("invalid escape");
                }
                result.push_back(text_[position_++]);
                continue;
            }
            result.push_back(ch);
        }
        throw std::runtime_error("unterminated string");
    }

    int parse_number() {
        const std::size_t start = position_;
        if (text_[position_] == '-') {
            ++position_;
        }
        while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])) != 0) {
            ++position_;
        }
        return std::stoi(text_.substr(start, position_ - start));
    }

    void expect(char ch) {
        if (position_ >= text_.size() || text_[position_] != ch) {
            throw std::runtime_error("unexpected json character");
        }
        ++position_;
    }

    bool peek(char ch) const {
        return position_ < text_.size() && text_[position_] == ch;
    }

    void skip_whitespace() {
        while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_])) != 0) {
            ++position_;
        }
    }

    std::string text_;
    std::size_t position_ = 0;
};

struct CaseDef {
    std::string name;
    std::string line;
    int current_index = kDirectionRadius;
    std::map<std::string, int> expected_empty_state_counts;
    std::map<std::string, int> expected_filled_state_counts;
};

std::string read_text_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::vector<CaseDef> load_cases(const std::string& path) {
    const JsonValue root = JsonParser(read_text_file(path)).parse();
    if (root.type != JsonValue::Type::kArray) {
        throw std::runtime_error("case file root must be an array");
    }

    std::vector<CaseDef> cases;
    for (const JsonValue& item : root.array_value) {
        if (item.type != JsonValue::Type::kObject) {
            throw std::runtime_error("case entry must be an object");
        }

        CaseDef case_def;
        case_def.name = item.object_value.at("name").string_value;
        case_def.line = item.object_value.at("line").string_value;
        if (const auto current_index_it = item.object_value.find("current_index");
            current_index_it != item.object_value.end()) {
            case_def.current_index = current_index_it->second.number_value;
        }

        const JsonValue& expected_empty = item.object_value.at("expected_empty_state_counts");
        if (expected_empty.type != JsonValue::Type::kObject) {
            throw std::runtime_error("expected_empty_state_counts must be an object");
        }
        for (const auto& entry : expected_empty.object_value) {
            case_def.expected_empty_state_counts.emplace(entry.first, entry.second.number_value);
        }

        const JsonValue& expected_filled = item.object_value.at("expected_filled_state_counts");
        if (expected_filled.type != JsonValue::Type::kObject) {
            throw std::runtime_error("expected_filled_state_counts must be an object");
        }
        for (const auto& entry : expected_filled.object_value) {
            case_def.expected_filled_state_counts.emplace(entry.first, entry.second.number_value);
        }
        cases.push_back(case_def);
    }
    return cases;
}

std::array<RelativeCellState, kDirectionSideCount> line_to_side_states(const std::string& line, int current_index) {
    if (line.size() != static_cast<std::size_t>(kDirectionLineLength)) {
        throw std::runtime_error("line length must be 11");
    }
    if (current_index < 0 || current_index >= kDirectionLineLength) {
        throw std::runtime_error("current_index must be inside 11-cell line");
    }

    std::array<RelativeCellState, kDirectionSideCount> states{};
    auto decode = [](char ch) {
        switch (ch) {
            case '_':
                return RelativeCellState::kEmpty;
            case 'X':
                return RelativeCellState::kSelf;
            case 'O':
                return RelativeCellState::kOther;
            default:
                throw std::runtime_error("unsupported line symbol");
        }
    };

    int side_index = 0;
    for (int index = 0; index < kDirectionLineLength; ++index) {
        if (index == current_index) {
            continue;
        }
        states[static_cast<std::size_t>(side_index++)] = decode(line[static_cast<std::size_t>(index)]);
    }
    return states;
}

std::string restore_line(const std::string& line, int current_index, bool fill_current_stone) {
    std::string restored = line;
    restored[static_cast<std::size_t>(current_index)] = fill_current_stone ? 'X' : '_';
    return restored;
}

std::map<std::string, int> non_zero_state_counts(const StateValueCountVector& counts) {
    std::map<std::string, int> result;
    for (const StateValueCount& item : counts) {
        if (item.count > 0) {
            result.emplace(std::string(state_value_name(item.state_id)), item.count);
        }
    }
    return result;
}

std::string map_to_json(const std::map<std::string, int>& values) {
    std::ostringstream output;
    output << "{";
    bool first = true;
    for (const auto& entry : values) {
        if (!first) {
            output << ", ";
        }
        first = false;
        output << "\"" << entry.first << "\": " << entry.second;
    }
    output << "}";
    return output.str();
}

std::string build_report(const std::vector<CaseDef>& cases, bool* all_passed) {
    std::ostringstream report;
    report << "[\n";
    bool first_case = true;
    *all_passed = true;

    for (const CaseDef& case_def : cases) {
        const auto side_states = line_to_side_states(case_def.line, case_def.current_index);
        const std::map<std::string, int> actual_empty = non_zero_state_counts(
            classify_restored_direction_state_counts(side_states, case_def.current_index, false)
        );
        const std::map<std::string, int> actual_filled = non_zero_state_counts(
            classify_restored_direction_state_counts(side_states, case_def.current_index, true)
        );
        const bool passed =
            actual_empty == case_def.expected_empty_state_counts &&
            actual_filled == case_def.expected_filled_state_counts;
        *all_passed = *all_passed && passed;

        if (!first_case) {
            report << ",\n";
        }
        first_case = false;

        report << "  {\n";
        report << "    \"name\": \"" << case_def.name << "\",\n";
        report << "    \"line\": \"" << case_def.line << "\",\n";
        report << "    \"current_index\": " << case_def.current_index << ",\n";
        report << "    \"empty_line\": \"" << restore_line(case_def.line, case_def.current_index, false) << "\",\n";
        report << "    \"filled_line\": \"" << restore_line(case_def.line, case_def.current_index, true) << "\",\n";
        report << "    \"passed\": " << (passed ? "true" : "false") << ",\n";
        report << "    \"expected_empty_state_counts\": " << map_to_json(case_def.expected_empty_state_counts) << ",\n";
        report << "    \"actual_empty_state_counts\": " << map_to_json(actual_empty) << ",\n";
        report << "    \"expected_filled_state_counts\": " << map_to_json(case_def.expected_filled_state_counts) << ",\n";
        report << "    \"actual_filled_state_counts\": " << map_to_json(actual_filled) << "\n";
        report << "  }";
    }

    report << "\n]\n";
    return report.str();
}

void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    output << text;
}

}  // namespace

int run_direction_pattern_cases(
    const std::string& input_path = "d:/code/gomoku_ai/cpp/test/direction_pattern_cases.json",
    const std::string& output_path = "d:/code/gomoku_ai/cpp/test/direction_pattern_test_output.json"
) {
    bool all_passed = false;
    const std::vector<CaseDef> cases = load_cases(input_path);
    const std::string report = build_report(cases, &all_passed);
    write_text_file(output_path, report);
    return all_passed ? 0 : 1;
}

}  // namespace gomoku::test

int main() {
    try {
        return gomoku::test::run_direction_pattern_cases();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 2;
    }
}
