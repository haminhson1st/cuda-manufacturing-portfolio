#pragma once

// Minimal "--name value" command-line parser. Every option takes exactly one value. Unknown,
// repeated, or malformed options are errors, so a typo cannot silently change an experiment.

#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace common {

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string name(argv[i]);
            if (name == "--help" || name == "-h") {
                help_ = true;
                continue;
            }
            if (name.size() < 3 || name.compare(0, 2, "--") != 0) {
                throw std::invalid_argument("Expected an option such as --name value, got '" + name + "'.");
            }
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + name + '.');
            if (!values_.emplace(name, argv[++i]).second) {
                throw std::invalid_argument("Option given twice: " + name + '.');
            }
        }
    }

    bool help() const { return help_; }

    // Uses `value` for `name` unless the command line already gave one (configuration files).
    void set_default(const std::string& name, const std::string& value) { values_.emplace(name, value); }

    std::string text(const std::string& name, const std::string& fallback) {
        used_.insert(name);
        const auto found = values_.find(name);
        return found == values_.end() ? fallback : found->second;
    }

    std::string choice(const std::string& name, const std::string& fallback,
                       std::initializer_list<const char*> allowed) {
        const std::string value = text(name, fallback);
        for (const char* option : allowed) {
            if (value == option) return value;
        }
        std::string message = name + " must be one of:";
        for (const char* option : allowed) message += std::string(" ") + option;
        throw std::invalid_argument(message + " (got '" + value + "').");
    }

    // Integer in [low, high]. Digits only, so "-1", "12x", and "1e3" are rejected.
    std::size_t count(const std::string& name, std::size_t fallback, std::size_t low, std::size_t high) {
        used_.insert(name);
        std::size_t value = fallback;
        const auto found = values_.find(name);
        if (found != values_.end()) {
            const std::string& s = found->second;
            if (s.empty() || s.size() > 18 || s.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument(name + " must be a non-negative integer (got '" + s + "').");
            }
            value = static_cast<std::size_t>(std::stoull(s));
        }
        if (value < low || value > high) {
            throw std::invalid_argument(name + " must be in [" + std::to_string(low) + ", "
                                        + std::to_string(high) + "] (got " + std::to_string(value) + ").");
        }
        return value;
    }

    // Finite floating-point number.
    double number(const std::string& name, double fallback) {
        used_.insert(name);
        const auto found = values_.find(name);
        if (found == values_.end()) return fallback;
        const std::string& s = found->second;
        std::size_t used = 0;
        double value = 0.0;
        try {
            value = std::stod(s, &used);
        } catch (const std::exception&) {
            used = 0;
        }
        if (used == 0 || used != s.size() || !std::isfinite(value)) {
            throw std::invalid_argument(name + " must be a finite number (got '" + s + "').");
        }
        return value;
    }

    // Call after reading every option: rejects options the program does not know.
    void finish() const {
        for (const auto& item : values_) {
            if (used_.count(item.first) == 0) throw std::invalid_argument("Unknown option " + item.first + '.');
        }
    }

private:
    std::map<std::string, std::string> values_;
    std::set<std::string> used_;
    bool help_ = false;
};

// Reads "name = value" lines (blank lines and # comments allowed) and applies each one as the
// default for --name, so options given on the command line still win. Unknown names are caught
// later by Args::finish(), exactly like a mistyped option.
inline void apply_config_file(Args& args, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("Cannot open configuration file " + path + '.');
    const auto trim = [](std::string s) {
        s.erase(0, s.find_first_not_of(" \t"));
        s.erase(s.find_last_not_of(" \t") + 1);
        return s;
    };
    std::set<std::string> seen;
    std::string line;
    std::size_t number = 0;
    while (std::getline(in, line)) {
        ++number;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line = trim(line);
        if (line.empty()) continue;
        const std::string where = path + ':' + std::to_string(number);
        const auto equals = line.find('=');
        if (equals == std::string::npos) throw std::invalid_argument(where + ": expected name = value.");
        const std::string name = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (name.empty() || value.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos) {
            throw std::invalid_argument(where + ": expected a lowercase option name and a value.");
        }
        if (name == "config") throw std::invalid_argument(where + ": configuration files cannot include other files.");
        if (!seen.insert(name).second) throw std::invalid_argument(where + ": " + name + " is set twice.");
        args.set_default("--" + name, value);
    }
}

}  // namespace common
