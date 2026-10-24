#pragma once

// CUDA helpers shared by Labs 02+: error checking, an owning device buffer, and event timing.
// Lab 01 keeps its own minimal version so the first program stays self-contained.

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace common {

inline void cuda_check(cudaError_t result, const char* call, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(call) + " (" + file + ':' + std::to_string(line)
                                 + "): " + cudaGetErrorString(result));
    }
}

}  // namespace common

#define CUDA_CHECK(call) ::common::cuda_check((call), #call, __FILE__, __LINE__)

// A launch reports configuration errors immediately. Errors raised while the kernel runs
// surface at the next synchronizing call, which is also checked.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace common {

// Owns `count` elements of device memory. Move-only, so ownership stays explicit.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()));
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);  // destructors must not throw
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) cudaFree(data_);
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* get() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void upload(const std::vector<T>& host) {
        if (host.size() != count_) throw std::logic_error("DeviceBuffer::upload: size mismatch.");
        if (count_ > 0) CUDA_CHECK(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        host.resize(count_);
        if (count_ > 0) CUDA_CHECK(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost));
    }
    // Sets every byte, e.g. 0xFF so floats that a kernel never writes read back as NaN.
    void fill_bytes(int value) {
        if (count_ > 0) CUDA_CHECK(cudaMemset(data_, value, bytes()));
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Page-locked ("pinned") host memory from cudaMallocHost. Copies from it can run asynchronously;
// copies from ordinary pageable memory cannot. Move-only, like DeviceBuffer.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) cudaFreeHost(data_);
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) cudaFreeHost(data_);
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* get() const { return data_; }
    std::size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// An owned CUDA stream (cudaStreamCreate: a blocking stream, which synchronizes with the legacy
// default stream).
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

// An owned CUDA event. Default flags keep timing enabled, so cudaEventElapsedTime works on it.
class Event {
public:
    explicit Event(unsigned int flags = cudaEventDefault) { CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// Milliseconds between two completed events.
inline double event_ms(const Event& start, const Event& stop) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    return milliseconds;
}

// Times device work between two events recorded in the same stream.
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        const cudaError_t result = cudaEventCreate(&stop_);
        if (result != cudaSuccess) {
            cudaEventDestroy(start_);
            CUDA_CHECK(result);
        }
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Records the stop event, waits for it, and returns the elapsed milliseconds.
    double stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// Runs work() `warmups` times, then times `reps` repetitions with CUDA events.
// work() must enqueue device work and check its launches with CUDA_CHECK_LAUNCH().
template <typename Work>
std::vector<double> time_device(int warmups, int reps, Work&& work) {
    for (int i = 0; i < warmups; ++i) work();
    CUDA_CHECK(cudaDeviceSynchronize());  // surfaces execution errors from the warm-up
    EventTimer timer;
    std::vector<double> times;
    for (int i = 0; i < reps; ++i) {
        timer.start();
        work();
        times.push_back(timer.stop());
    }
    return times;
}

}  // namespace common
