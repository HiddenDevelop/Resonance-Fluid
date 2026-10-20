#include "FluidSimulation.cuh"
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace {
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
}
int main() {
    int count=0; auto availability=cudaGetDeviceCount(&count);
    if(availability==cudaErrorNoDevice || availability==cudaErrorInsufficientDriver || (availability==cudaSuccess && count==0)) {
        std::cout<<"SKIP: CUDA GPU/driver unavailable\n"; return 77;
    }
    try {
        checkCuda(availability,"enumerate CUDA devices");
        FluidSimulation fluid(32);
        fluid.step(1.0f/120);
        auto zero=fluid.diagnostics();
        require(zero.finite && zero.maxSpeed==0 && zero.dyeSum==0 && zero.divergenceRms==0,"Zero state must stay zero");
        Splat s; s.radius=3; s.velocityX=35; s.velocityY=-18;
        fluid.addSplat(s);
        auto before=fluid.diagnostics();
        fluid.settings.pressureIterations=1000; fluid.project();
        auto after=fluid.diagnostics();
        std::cout<<"Projection divergence RMS: "<<before.divergenceRms<<" -> "<<after.divergenceRms<<'\n';
        require(before.divergenceRms>0 && after.divergenceRms<0.1*before.divergenceRms,"Projection must remove at least 90% of test divergence");
        require(after.finite && after.wallsClosed,"Projection must preserve finite fields and closed walls");
        fluid.reset(); s.velocityX=s.velocityY=0; fluid.addSplat(s);
        before=fluid.diagnostics();
        constexpr float dt=1.0f/120;
        fluid.step(dt); after=fluid.diagnostics();
        double expected=before.dyeSum*std::exp(-fluid.settings.dyeDecay*dt);
        require(std::abs(after.dyeSum-expected)<expected*1e-5,"Stationary dye must follow exponential decay");
        // Exercise the same output kernel used by interop without requiring an OpenGL context.
        DeviceBuffer<float4> pixels(32*32); fluid.writePixels(pixels.data());
        float4 center{};
        checkCuda(cudaMemcpy(&center,pixels.data()+16*32+16,sizeof(center),cudaMemcpyDeviceToHost),"read test pixel");
        require(center.z>0 && center.w==1,"Output kernel must write dye and opaque alpha");
        for(int i=0;i<300;++i) {
            if(i%10==0) {s.x=0.1f+0.8f*(i%70)/70.0f; s.velocityX=20; s.velocityY=10; fluid.addSplat(s);}
            fluid.step(dt);
        }
        after=fluid.diagnostics();
        require(after.finite && after.wallsClosed && after.maxSpeed>0,"Interactive stress test must remain finite with closed walls");
        fluid.reset(); after=fluid.diagnostics();
        require(after.dyeSum==0 && after.maxSpeed==0 && after.divergenceRms==0,"Reset must clear fields");
        // A non-multiple-of-16 extent catches missing CUDA launch bounds guards.
        FluidSimulation uneven(31); uneven.addSplat(s); uneven.step(dt);
        after=uneven.diagnostics(); require(after.finite && after.wallsClosed,"Odd grid must remain valid");
        bool rejected=false; try {fluid.step(0);} catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"Invalid timestep must be rejected");
        checkCuda(cudaDeviceSynchronize(),"finish numerical tests");
        std::cout<<"PASS: CUDA numerical regression tests\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}