#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <Windows.h>
#include "FluidSimulation.cuh"
#include "Renderer.h"
#include "Input.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace
{
    struct GlfwLifetime
    {
        GlfwLifetime()
        {
            glfwSetErrorCallback([](int code, const char *text)
                                 { std::cerr << "GLFW " << code << ": " << text << '\n'; });
            if (!glfwInit())
                throw std::runtime_error("GLFW initialization failed");
        }
        ~GlfwLifetime() { glfwTerminate(); }
    };
    std::filesystem::path executableDirectory()
    {
        std::wstring path(32768, L'\0');
        DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
        if (length == 0 || length >= path.size())
            throw std::runtime_error("Cannot locate executable directory");
        path.resize(length);
        return std::filesystem::path(path).parent_path();
    }
    void selectGraphicsDevice()
    {
        unsigned count = 0;
        int devices[8]{};
        auto result = cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListCurrentFrame);
        if (result != cudaSuccess || count == 0)
            throw std::runtime_error(std::string("No CUDA device shares this OpenGL context. Select the NVIDIA high-performance GPU in Windows Graphics settings. CUDA: ") + cudaGetErrorString(result));
        checkCuda(cudaSetDevice(devices[0]), "select graphics CUDA device");
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, devices[0]), "query GPU");
        std::cout << "CUDA GPU: " << properties.name << "\nOpenGL: " << glGetString(GL_RENDERER) << '\n';
    }
}
int main(int argc, char **argv)
{
    try
    {
        bool smoke = argc == 2 && std::string(argv[1]) == "--smoke-test";
        if (argc > 1 && !smoke)
        {
            std::cerr << "Usage: Resonance [--smoke-test]\n";
            return 2;
        }
        GlfwLifetime glfw;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
            glfwCreateWindow(1000, 900, "Resonance", nullptr, nullptr), glfwDestroyWindow);
        if (!window)
            throw std::runtime_error("OpenGL 3.3 window creation failed");
        // Retain short key taps even if press and release arrive in the same poll.
        glfwSetInputMode(window.get(), GLFW_STICKY_KEYS, GLFW_TRUE);
        glfwMakeContextCurrent(window.get());
        if (!gladLoadGL(glfwGetProcAddress))
            throw std::runtime_error("OpenGL function loading failed");
        glfwSwapInterval(1);
        selectGraphicsDevice();
        // Reverse destruction order: unregister/delete renderer resources before destroying the GL context.
        FluidSimulation simulation(256);
        Renderer renderer(simulation.size(), executableDirectory() / "shaders");
        Input input;
        std::cout << "LMB drag: dye + force | Space: pause | R: reset | C: color | Up/Down: dye decay | Esc: quit\n";
        constexpr double fixedStep = 1.0 / 120.0;
        constexpr int maxCatchUpSteps = 8;
        double previous = glfwGetTime(), accumulator = 0, titleStart = previous;
        int frames = 0, smokeFrames = 0;
        bool paused = false, dirty = true;
        const double smokeStart = glfwGetTime();
        while (!glfwWindowShouldClose(window.get()))
        {
            glfwPollEvents();
            double now = glfwGetTime(), elapsed = std::clamp(now - previous, 0.0, 0.25);
            previous = now;
            int width, height;
            glfwGetFramebufferSize(window.get(), &width, &height);
            input.poll(window.get(), elapsed, simulation.size());
            if (input.pausePressed)
            {
                paused = !paused;
                accumulator = 0;
                input.discardMotion();
            }
            if (input.resetPressed)
            {
                simulation.reset();
                accumulator = 0;
                input.discardMotion();
                dirty = true;
            }
            simulation.settings.dyeDecay = std::clamp(simulation.settings.dyeDecay + input.decayAdjustment, 0.0f, 1.0f);
            if (width == 0 || height == 0 || glfwGetWindowAttrib(window.get(), GLFW_ICONIFIED))
            {
                accumulator = 0;
                input.discardMotion();
                glfwWaitEventsTimeout(0.05);
                previous = glfwGetTime();
                continue;
            }
            if (smoke)
            {
                Splat s;
                s.x = 0.5f + 0.18f * std::cos(smokeFrames * 0.05f);
                s.y = 0.5f + 0.18f * std::sin(smokeFrames * 0.05f);
                s.velocityX = -12 * std::sin(smokeFrames * 0.05f);
                s.velocityY = 12 * std::cos(smokeFrames * 0.05f);
                simulation.addSplat(s);
                simulation.step(float(fixedStep));
                simulation.step(float(fixedStep));
                dirty = true;
            }
            else if (!paused)
            {
                accumulator += elapsed;
                int steps = 0;
                while (accumulator >= fixedStep && steps < maxCatchUpSteps)
                {
                    input.inject(simulation, float(fixedStep));
                    simulation.step(float(fixedStep));
                    accumulator -= fixedStep;
                    ++steps;
                    dirty = true;
                }
                // Deliberately lose wall-clock time under overload; never take a giant solver step.
                if (accumulator >= fixedStep)
                    accumulator = std::fmod(accumulator, fixedStep);
            }
            else
            {
                accumulator = 0;
                input.discardMotion();
            }
            if (dirty)
            {
                renderer.upload(simulation);
                dirty = false;
            }
            renderer.draw(width, height);
            glfwSwapBuffers(window.get());
            ++frames;
            if (now - titleStart >= 0.5)
            {
                auto title = std::string("Resonance | ") + (paused ? "PAUSED | " : "") + std::to_string(int(frames / (now - titleStart))) +
                             " FPS | Space pause | R reset | C color | Dye decay " + std::to_string(simulation.settings.dyeDecay);
                glfwSetWindowTitle(window.get(), title.c_str());
                frames = 0;
                titleStart = now;
            }
            if (smoke && ++smokeFrames >= 240)
            {
                checkCuda(cudaDeviceSynchronize(), "finish smoke test");
                auto d = simulation.diagnostics();
                if (!d.finite || !d.wallsClosed || d.dyeSum <= 0 || glGetError() != GL_NO_ERROR)
                    throw std::runtime_error("Smoke test failed: invalid fields or OpenGL error");
                std::cout << "PASS: 480 CUDA steps and 240 CUDA/OpenGL presentations; divergence RMS " << d.divergenceRms
                          << "; average presentation rate " << 240 / (glfwGetTime() - smokeStart) << " FPS (includes synchronization, vsync and final diagnostics)\n";
                break;
            }
        }
        if (smoke && smokeFrames < 240)
            throw std::runtime_error("Smoke test interrupted before completion");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Resonance: " << error.what() << '\n';
        return 1;
    }
}