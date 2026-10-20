#pragma once
#include <filesystem>
struct cudaGraphicsResource;
class FluidSimulation;

class Renderer
{
public:
    Renderer(int gridSize, const std::filesystem::path &shaderDirectory);
    ~Renderer();
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;
    void upload(const FluidSimulation &simulation);
    void draw(int framebufferWidth, int framebufferHeight);

private:
    void release() noexcept;
    int size_;
    unsigned int program_ = 0, vao_ = 0, texture_ = 0, pixelBuffer_ = 0;
    cudaGraphicsResource *resource_ = nullptr;
};