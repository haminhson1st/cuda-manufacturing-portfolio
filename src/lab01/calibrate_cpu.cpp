#include "common.hpp"

int main(int argc, char** argv) {
    try {
        const auto n = lab01::parse_count(argc, argv);
        const auto x = lab01::make_input(n);
        std::vector<float> y(n);
        for (int i = 0; i < lab01::warmups; ++i) lab01::calibrate_cpu(x, y);
        std::vector<double> times;
        for (int i = 0; i < lab01::repetitions; ++i) {
            const auto start = lab01::Clock::now();
            lab01::calibrate_cpu(x, y);
            const auto stop = lab01::Clock::now();
            times.push_back(lab01::elapsed_ms(start, stop));
        }
        const auto check = lab01::validate(x, y);
        if (argc == 3) lab01::write_observations(argv[2], {{"cpu_ms", times}});
        lab01::print_header();
        std::cout << "cpu," << n << ',' << lab01::repetitions << ','
                  << lab01::median(times) << ",,," << check.max_abs_error << ','
                  << check.correct << '\n';
        return check.correct ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
