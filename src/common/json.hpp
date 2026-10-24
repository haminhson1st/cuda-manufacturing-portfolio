#pragma once

// Minimal JSON writing for run records (Labs 11 and 12): escaped strings, finite numbers, arrays,
// and objects built from already-encoded values. Writing only; nothing here parses JSON.

#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace common {

inline std::string json_text(const std::string& text) {
    std::string out = "\"";
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += raw;
        } else if (c < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", c);
            out += escaped;
        } else {
            out += raw;
        }
    }
    return out + '"';
}

// Non-finite values (NaN for "not set", infinity for "disabled") become null.
inline std::string json_number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

inline std::string json_array(const std::vector<std::string>& encoded) {
    std::string out = "[";
    for (std::size_t i = 0; i < encoded.size(); ++i) out += (i ? ", " : "") + encoded[i];
    return out + "]";
}

// An object whose values are already-encoded JSON. str(indent) indents fields by `indent`
// and the closing brace by two spaces less, so nested objects line up.
struct JsonObject {
    std::vector<std::pair<std::string, std::string>> fields;

    JsonObject& add(const std::string& key, const std::string& encoded) {
        fields.emplace_back(key, encoded);
        return *this;
    }

    std::string str(const std::string& indent = "  ") const {
        std::string out = "{\n";
        for (std::size_t i = 0; i < fields.size(); ++i) {
            out += indent + json_text(fields[i].first) + ": " + fields[i].second + (i + 1 < fields.size() ? ",\n" : "\n");
        }
        return out + indent.substr(2) + "}";
    }
};

inline std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    char text[32];
    std::strftime(text, sizeof text, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    return text;
}

}  // namespace common
