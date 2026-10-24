#include <cuda_runtime.h>
#include "common.hpp"

// Every CUDA API and launch is checked. On a fatal error this short example exits;
// a reusable application should wrap owned buffers/events in RAII types.
inline void cuda_check(cudaError_t result, const char* call, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(call) + " at line " + std::to_string(line)
                                 + ": " + cudaGetErrorString(result));
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call, __LINE__)

__global__ void calibrate(const float* x, float* y, std::size_t n, float gain, float offset) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gain * x[i] + offset;
}

int main(int argc, char** argv) {
    try {
        const auto n = lab01::parse_count(argc, argv);
        const auto x = lab01::make_input(n);
        std::vector<float> y(n);
        const auto bytes = n * sizeof(float);
        float* device_x = nullptr;
        float* device_y = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_x), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_y), bytes));
        constexpr int threads = 256;
        const int blocks = static_cast<int>((n + threads - 1) / threads);

        // Warm up the whole transfer/compute sequence before steady-state timing.
        for (int i = 0; i < lab01::warmups; ++i) {
            CUDA_CHECK(cudaMemcpy(device_x, x.data(), bytes, cudaMemcpyHostToDevice));
            calibrate<<<blocks, threads>>>(device_x, device_y, n, lab01::gain, lab01::offset);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(y.data(), device_y, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        std::vector<double> kernel_times;
        for (int i = 0; i < lab01::repetitions; ++i) {
            CUDA_CHECK(cudaEventRecord(start_event));
            calibrate<<<blocks, threads>>>(device_x, device_y, n, lab01::gain, lab01::offset);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
            kernel_times.push_back(milliseconds);
        }

        // This boundary includes copies and completion, but NOT allocation or I/O.
        // Pageable host buffers are intentional; pinned buffers are a later lab.
        std::vector<double> transfer_times;
        for (int i = 0; i < lab01::repetitions; ++i) {
            const auto start = lab01::Clock::now();
            CUDA_CHECK(cudaMemcpy(device_x, x.data(), bytes, cudaMemcpyHostToDevice));
            calibrate<<<blocks, threads>>>(device_x, device_y, n, lab01::gain, lab01::offset);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(y.data(), device_y, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto stop = lab01::Clock::now();
            transfer_times.push_back(lab01::elapsed_ms(start, stop));
        }

        const auto check = lab01::validate(x, y);
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        CUDA_CHECK(cudaFree(device_x));
        CUDA_CHECK(cudaFree(device_y));
        if (argc == 3) {
            lab01::write_observations(argv[2], {
                {"kernel_ms", kernel_times}, {"transfer_compute_ms", transfer_times}});
        }
        lab01::print_header();
        std::cout << "cuda," << n << ',' << lab01::repetitions << ",,"
                  << lab01::median(kernel_times) << ',' << lab01::median(transfer_times)
                  << ',' << check.max_abs_error << ',' << check.correct << '\n';
        return check.correct ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
