#pragma once
#include "FluidSimulation.cuh"
struct GLFWwindow;
class Input
{
public:
    void poll(GLFWwindow *window, double frameSeconds, int gridSize);
    void inject(FluidSimulation &simulation, float dt);
    void discardMotion() { pendingX_ = pendingY_ = 0; }
    bool pausePressed = false, resetPressed = false, colorPressed = false;
    float decayAdjustment = 0;

private:
    bool previousPause_ = false, previousReset_ = false, previousColor_ = false, dragging_ = false;
    float x_ = 0, y_ = 0, previousX_ = 0, previousY_ = 0, pendingX_ = 0, pendingY_ = 0;
    int color_ = 0;
};