#pragma once

// 8-bit images with an explicit row pitch (bytes from one row start to the next), synthetic
// test patterns, and PGM/PPM input and output. Shared by Labs 05–07.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace common {

// Non-owning views passed to kernels by value. `pitch` is in bytes and may exceed
// width * channels; `width` and `height` count pixels.
struct ImageView {
    std::uint8_t* data;
    std::size_t pitch;
    int width;
    int height;
};

struct ConstImageView {
    const std::uint8_t* data;
    std::size_t pitch;
    int width;
    int height;
};

// Host image whose rows may carry `pad` extra bytes after width * channels.
struct HostImage {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::size_t pitch = 0;
    std::vector<std::uint8_t> bytes;

    HostImage() = default;
    HostImage(int w, int h, int c, std::size_t pad = 0, std::uint8_t fill = 0)
        : width(w), height(h), channels(c),
          pitch(static_cast<std::size_t>(w) * static_cast<std::size_t>(c) + pad),
          bytes(pitch * static_cast<std::size_t>(h), fill) {}

    std::size_t row_bytes() const { return static_cast<std::size_t>(width) * static_cast<std::size_t>(channels); }
    std::uint8_t* row(int y) { return bytes.data() + static_cast<std::size_t>(y) * pitch; }
    const std::uint8_t* row(int y) const { return bytes.data() + static_cast<std::size_t>(y) * pitch; }
    std::uint8_t& at(int x, int y, int c = 0) { return row(y)[static_cast<std::size_t>(x) * channels + c]; }
    std::uint8_t at(int x, int y, int c = 0) const { return row(y)[static_cast<std::size_t>(x) * channels + c]; }
    ConstImageView view() const { return {bytes.data(), pitch, width, height}; }
};

inline std::uint32_t hash3(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    std::uint32_t h = x * 0x8DA6B343u ^ y * 0xD8163841u ^ z * 0xCB1AB31Fu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

// Deterministic content for (x, y, channel). Patterns: constant, gradient, checker, noise,
// impulse (one bright pixel at the center), corner (one bright pixel at (0, 0)).
inline std::uint8_t pattern_value(const std::string& pattern, int x, int y, int c, int width, int height) {
    if (pattern == "constant") return static_cast<std::uint8_t>(100 + 30 * c);
    if (pattern == "gradient") {
        return static_cast<std::uint8_t>((x * 200 / std::max(1, width - 1) + y * 55 / std::max(1, height - 1)
                                          + 20 * c) % 256);
    }
    if (pattern == "checker") return static_cast<std::uint8_t>(((x / 8 + y / 8) % 2) ? 200 - 10 * c : 40 + 10 * c);
    if (pattern == "noise") {
        return static_cast<std::uint8_t>(hash3(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
                                               static_cast<std::uint32_t>(c)) & 0xFFu);
    }
    if (pattern == "impulse") return (x == width / 2 && y == height / 2) ? 255 : 0;
    if (pattern == "corner") return (x == 0 && y == 0) ? 255 : 0;
    throw std::invalid_argument("Unknown image pattern '" + pattern + "'.");
}

// Fills every pixel; padding bytes keep `pad_fill`, so code that reads padding as pixels is caught.
inline HostImage make_pattern(const std::string& pattern, int width, int height, int channels,
                              std::size_t pad = 0, std::uint8_t pad_fill = 0xEE) {
    HostImage image(width, height, channels, pad, pad_fill);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) image.at(x, y, c) = pattern_value(pattern, x, y, c, width, height);
        }
    }
    return image;
}

// Number of pixel bytes that differ (padding ignored); `first` describes the first difference.
inline std::size_t count_differences(const HostImage& expected, const HostImage& actual, std::string& first) {
    if (expected.width != actual.width || expected.height != actual.height || expected.channels != actual.channels) {
        throw std::logic_error("count_differences: image shapes differ.");
    }
    std::size_t differences = 0;
    for (int y = 0; y < expected.height; ++y) {
        for (int x = 0; x < expected.width; ++x) {
            for (int c = 0; c < expected.channels; ++c) {
                if (expected.at(x, y, c) == actual.at(x, y, c)) continue;
                if (differences++ == 0) {
                    first = "x=" + std::to_string(x) + " y=" + std::to_string(y) + " channel=" + std::to_string(c)
                            + ": expected " + std::to_string(expected.at(x, y, c)) + ", got "
                            + std::to_string(actual.at(x, y, c));
                }
            }
        }
    }
    return differences;
}

// Writes a 1-channel image as binary PGM (P5) or a 3-channel RGB image as PPM (P6).
inline void write_pnm(const std::string& path, const HostImage& image) {
    if (image.channels != 1 && image.channels != 3) throw std::invalid_argument("PNM output needs 1 or 3 channels.");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open " + path + '.');
    out << (image.channels == 1 ? "P5" : "P6") << '\n' << image.width << ' ' << image.height << "\n255\n";
    for (int y = 0; y < image.height; ++y) {
        out.write(reinterpret_cast<const char*>(image.row(y)), static_cast<std::streamsize>(image.row_bytes()));
    }
    if (!out) throw std::runtime_error("Failed writing " + path + '.');
}

// Reads a binary PGM (P5, gray) or PPM (P6, RGB) file with maxval 255 into a packed image.
// Header comments (#) are allowed; exactly one whitespace byte separates maxval from the pixels.
inline HostImage read_pnm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open image " + path + '.');
    const auto token = [&in]() {
        std::string text;
        char ch = 0;
        while (in.get(ch)) {
            if (ch == '#' && text.empty()) {
                std::string comment;
                std::getline(in, comment);
                continue;
            }
            if (std::isspace(static_cast<unsigned char>(ch))) {
                if (text.empty()) continue;
                break;  // consumes the single whitespace byte that ends the token
            }
            text += ch;
        }
        return text;
    };
    const std::string magic = token();
    if (magic != "P5" && magic != "P6") {
        throw std::runtime_error(path + ": only binary PGM (P5) and PPM (P6) images are supported.");
    }
    const auto number = [&](const char* what) {
        const std::string text = token();
        if (text.empty() || text.size() > 6 || text.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error(path + ": invalid " + what + " in the header.");
        }
        return std::stoi(text);
    };
    const int width = number("width");
    const int height = number("height");
    const int maxval = number("maxval");
    if (width < 1 || height < 1 || width > 16384 || height > 16384) {
        throw std::runtime_error(path + ": image size must be 1..16384 in each dimension.");
    }
    if (maxval != 255) throw std::runtime_error(path + ": only 8-bit images (maxval 255) are supported.");
    HostImage image(width, height, magic == "P5" ? 1 : 3);
    in.read(reinterpret_cast<char*>(image.bytes.data()), static_cast<std::streamsize>(image.bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(image.bytes.size())) {
        throw std::runtime_error(path + ": pixel data is truncated.");
    }
    return image;
}

// Parses a block shape such as "16x16".
inline void parse_block(const std::string& text, unsigned& x, unsigned& y) {
    const auto split = text.find('x');
    const std::string left = split == std::string::npos ? "" : text.substr(0, split);
    const std::string right = split == std::string::npos ? "" : text.substr(split + 1);
    const auto digits = [](const std::string& s) {
        return !s.empty() && s.size() <= 4 && s.find_first_not_of("0123456789") == std::string::npos;
    };
    if (!digits(left) || !digits(right)) throw std::invalid_argument("--block must look like 16x16 (got '" + text + "').");
    x = static_cast<unsigned>(std::stoul(left));
    y = static_cast<unsigned>(std::stoul(right));
    if (x == 0 || y == 0 || x * y > 1024) {
        throw std::invalid_argument("--block needs positive sides and at most 1024 threads (got '" + text + "').");
    }
}

}  // namespace common
