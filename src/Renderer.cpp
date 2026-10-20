#include <glad/gl.h>
#include "Renderer.h"
#include "FluidSimulation.cuh"
#include <cuda_gl_interop.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if(!file) throw std::runtime_error("Cannot read shader: " + path.string());
    std::ostringstream contents; contents << file.rdbuf(); return contents.str();
}
GLuint compile(GLenum type, const std::filesystem::path& path) {
    auto source=readFile(path); const char* text=source.c_str();
    GLuint shader=glCreateShader(type);
    glShaderSource(shader,1,&text,nullptr); glCompileShader(shader);
    GLint success=0; glGetShaderiv(shader,GL_COMPILE_STATUS,&success);
    if(!success) {
        GLint length=0; glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&length);
        std::vector<char> log(std::max(length,1)); glGetShaderInfoLog(shader,length,nullptr,log.data());
        glDeleteShader(shader); throw std::runtime_error(path.string()+": "+log.data());
    }
    return shader;
}
GLuint makeProgram(const std::filesystem::path& dir) {
    GLuint vertex=compile(GL_VERTEX_SHADER,dir/"fullscreen.vert"), fragment=0, program=0;
    try {
        fragment=compile(GL_FRAGMENT_SHADER,dir/"fluid.frag");
        program=glCreateProgram(); glAttachShader(program,vertex); glAttachShader(program,fragment);
        glLinkProgram(program);
        GLint success=0; glGetProgramiv(program,GL_LINK_STATUS,&success);
        if(!success) {
            GLint length=0; glGetProgramiv(program,GL_INFO_LOG_LENGTH,&length);
            std::vector<char> log(std::max(length,1)); glGetProgramInfoLog(program,length,nullptr,log.data());
            throw std::runtime_error(std::string("Shader link: ")+log.data());
        }
    } catch(...) { glDeleteShader(vertex); glDeleteShader(fragment); glDeleteProgram(program); throw; }
    glDeleteShader(vertex); glDeleteShader(fragment); return program;
}
}
Renderer::Renderer(int n,const std::filesystem::path& shaders) : size_(n) {
    try {
        program_=makeProgram(shaders);
        glGenVertexArrays(1,&vao_);
        glGenTextures(1,&texture_); glBindTexture(GL_TEXTURE_2D,texture_);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,n,n,0,GL_RGBA,GL_FLOAT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenBuffers(1,&pixelBuffer_); glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pixelBuffer_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER,GLsizeiptr(n)*n*sizeof(float4),nullptr,GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
        if(glGetError()!=GL_NO_ERROR) throw std::runtime_error("OpenGL resource allocation failed");
        checkCuda(cudaGraphicsGLRegisterBuffer(&resource_,pixelBuffer_,cudaGraphicsRegisterFlagsWriteDiscard),"register OpenGL pixel buffer");
        glUseProgram(program_); glUniform1i(glGetUniformLocation(program_,"fluidTexture"),0);
    } catch(...) { release(); throw; }
}
Renderer::~Renderer() { release(); }
void Renderer::release() noexcept {
    if(resource_) { cudaGraphicsUnregisterResource(resource_); resource_=nullptr; }
    glDeleteBuffers(1,&pixelBuffer_); glDeleteTextures(1,&texture_);
    glDeleteVertexArrays(1,&vao_); glDeleteProgram(program_);
}
void Renderer::upload(const FluidSimulation& simulation) {
    checkCuda(cudaGraphicsMapResources(1,&resource_),"map OpenGL pixel buffer");
    try {
        float4* pixels=nullptr; size_t bytes=0;
        checkCuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pixels),&bytes,resource_),"get pixel buffer pointer");
        if(bytes<size_t(size_)*size_*sizeof(float4)) throw std::runtime_error("Pixel buffer too small");
        simulation.writePixels(pixels);
    } catch(...) { cudaGraphicsUnmapResources(1,&resource_); throw; }
    checkCuda(cudaGraphicsUnmapResources(1,&resource_),"unmap OpenGL pixel buffer");
    // Offset zero into the PBO: this is a GPU-to-GPU upload, not a CPU pointer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pixelBuffer_); glBindTexture(GL_TEXTURE_2D,texture_);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,size_,size_,GL_RGBA,GL_FLOAT,nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
}
void Renderer::draw(int w,int h) {
    glViewport(0,0,w,h); glClearColor(0.002f,0.003f,0.008f,1); glClear(GL_COLOR_BUFFER_BIT);
    int side=std::min(w,h); glViewport((w-side)/2,(h-side)/2,side,side);
    glUseProgram(program_); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,texture_);
    glBindVertexArray(vao_); glDrawArrays(GL_TRIANGLES,0,3);
}
