#pragma once
#include "CudaUtils.cuh"
#include "StepTimer.cuh"
#include <memory>

struct Splat {
    // Normalized position, velocity change in cells/second, radius in cells.
    float x = 0.5f, y = 0.5f;
    float velocityX = 0, velocityY = 0;
    float radius = 7;
    float3 color = {0.08f, 0.7f, 1.0f};
    float dyeAmount = 0.3f;
};
struct FluidSettings {
    float velocityDecay = 0.15f; // exponential rate per second
    float dyeDecay = 0.08f;
    int pressureIterations = 80;
};
struct Diagnostics {
    double divergenceRms = 0;
    double dyeSum = 0;
    float maxSpeed = 0;
    bool finite = true;
    bool wallsClosed = true;
};

class FluidSimulation {
public:
    explicit FluidSimulation(int size = 256);
    void reset();
    void addSplat(const Splat& splat);
    void step(float dt);
    void project(); // also exposed for the numerical regression test
    void writePixels(float4* devicePixels) const;
    Diagnostics diagnostics() const; // slow CPU readback, tests only
    int size() const { return size_; }
    FluidSettings settings;
    std::unique_ptr<StepTimer> timing; // null during ordinary usage
private:
    int size_;
    // MAC grid. Both face fields use (N+1)^2 storage for simple indexing.
    DeviceBuffer<float> u_, uNext_, v_, vNext_;
    DeviceBuffer<float4> dye_, dyeNext_;
    DeviceBuffer<float> pressure_, pressureNext_, divergence_;
};
