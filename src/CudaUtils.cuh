#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

inline void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// Single-owner device allocation; destruction must never throw.
template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }
    size_t bytes() const { return count_ * sizeof(T); }
    void clear() { checkCuda(cudaMemset(data_, 0, bytes()), "clear device field"); }
    void swap(DeviceBuffer& other) noexcept { std::swap(data_, other.data_); std::swap(count_, other.count_); }
private:
    T* data_ = nullptr;
    size_t count_;
};
