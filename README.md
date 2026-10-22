# Resonance-Fluid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Resonance-Fluid is a real-time fluid simulation that reacts to sound. The fluid solver runs on the GPU with CUDA, audio is captured and analyzed on the CPU, and the result is drawn with OpenGL.

## Requirements

- Windows (audio capture uses the Windows COM audio APIs)
- NVIDIA GPU with a recent driver
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.0 or newer
- [CMake](https://cmake.org/download/) 3.24 or newer
- A C++20 compiler (Visual Studio 2022 recommended)
- Python 3 with Jinja2, which GLAD needs to generate its OpenGL loader

GLFW and GLAD are downloaded automatically by CMake during configuration, so there is nothing else to install.

## Setup

Clone the repository and install the Python dependency:

```
git clone https://github.com/HiddenDevelop/Resonance-Fluid.git
cd Resonance-Fluid
python -m pip install Jinja2==3.1.6
```

Configure and build:

```
cmake -B build
cmake --build build --config Release
```

Run the app:

```
build\Release\Resonance.exe
```

The build copies the `shaders` folder next to the executable, so run it from the build output rather than moving the `.exe` elsewhere on its own.

By default CMake targets the GPU installed in your machine. To build for a different one, pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` (for example `86`) when configuring.

## Tests

```
ctest --test-dir build -C Release
```

This runs the audio analysis tests, a Python reference check for the pressure projection step, and the GPU fluid tests. The GPU tests are skipped if no CUDA device is available.

## Project layout

```
src/       Fluid solver (CUDA), audio capture and analysis, renderer, input
shaders/   GLSL shaders
tests/     Audio, fluid and pressure projection tests
```

## License

Released under the [MIT License](LICENSE).
