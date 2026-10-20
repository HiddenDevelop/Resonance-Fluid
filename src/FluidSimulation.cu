#include "FluidSimulation.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr int blockSide = 16;
dim3 blocks(int extent) { return dim3((extent + blockSide - 1) / blockSide, (extent + blockSide - 1) / blockSide); }
const dim3 threads(blockSide, blockSide);
size_t cellCount(int n) {
    if (n < 8 || n > 2048) throw std::invalid_argument("Grid size must be between 8 and 2048");
    return size_t(n) * n;
}
size_t faceCount(int n) { cellCount(n); return size_t(n + 1) * (n + 1); }
__device__ float mixValue(float a, float b, float t) { return a + t * (b - a); }
__device__ float4 mixValue(float4 a, float4 b, float t) {
    return make_float4(mixValue(a.x,b.x,t), mixValue(a.y,b.y,t), mixValue(a.z,b.z,t), 0);
}
// Sample coordinates are relative to the field's own lattice origin.
template<class T> __device__ T sample(const T* field, float x, float y, int width, int height, int stride) {
    x = fminf(fmaxf(x, 0.0f), float(width - 1));
    y = fminf(fmaxf(y, 0.0f), float(height - 1));
    int x0 = int(x), y0 = int(y);
    int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    return mixValue(mixValue(field[y0*stride+x0], field[y0*stride+x1], x-x0),
                    mixValue(field[y1*stride+x0], field[y1*stride+x1], x-x0), y-y0);
}
__device__ float2 velocity(const float* u, const float* v, float x, float y, int n) {
    return make_float2(sample(u,x,y-0.5f,n+1,n,n+1), sample(v,x-0.5f,y,n,n+1,n+1));
}
__device__ float weight(float x, float y, Splat s, int n) {
    float dx=x-s.x*n, dy=y-s.y*n;
    return expf(-(dx*dx+dy*dy)/(2*s.radius*s.radius));
}
__global__ void splatKernel(float* u, float* v, float4* dye, int n, Splat s) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>n || y>n) return;
    int f=y*(n+1)+x;
    if(x>0 && x<n && y<n) u[f]+=s.velocityX*weight(float(x),y+0.5f,s,n);
    if(y>0 && y<n && x<n) v[f]+=s.velocityY*weight(x+0.5f,float(y),s,n);
    if(x<n && y<n) {
        float w=s.dyeAmount*weight(x+0.5f,y+0.5f,s,n);
        float4 d=dye[y*n+x];
        dye[y*n+x]=make_float4(d.x+s.color.x*w,d.y+s.color.y*w,d.z+s.color.z*w,0);
    }
}
__global__ void advectVelocity(const float* u,const float* v,float* outU,float* outV,int n,float dt,float decay) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>n || y>n) return;
    int i=y*(n+1)+x;
    outU[i]=0; outV[i]=0; // normal wall velocity and unused padding stay zero
    if(x>0 && x<n && y<n) {
        float2 a=velocity(u,v,float(x),y+0.5f,n);
        outU[i]=decay*sample(u,x-dt*a.x,y-dt*a.y,n+1,n,n+1);
    }
    if(y>0 && y<n && x<n) {
        float2 a=velocity(u,v,x+0.5f,float(y),n);
        outV[i]=decay*sample(v,x-dt*a.x,y-dt*a.y,n,n+1,n+1);
    }
}
__global__ void divergenceKernel(const float* u,const float* v,float* div,int n) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=n || y>=n) return;
    int i=y*(n+1)+x;
    div[y*n+x]=u[i+1]-u[i]+v[i+n+1]-v[i];
}
__global__ void pressureKernel(const float* p,float* next,const float* div,int n) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=n || y>=n) return;
    int i=y*n+x, count=0; float sum=0;
    // Neumann walls: only interior neighbors contribute. No flow through walls.
    if(x>0) {sum+=p[i-1]; ++count;} if(x<n-1) {sum+=p[i+1]; ++count;}
    if(y>0) {sum+=p[i-n]; ++count;} if(y<n-1) {sum+=p[i+n]; ++count;}
    // Weighted Jacobi damps the alternating mode of the Neumann system.
    next[i]=mixValue(p[i],(sum-div[i])/count,2.0f/3.0f);
}
__global__ void gradientKernel(float* u,float* v,const float* p,int n) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>n || y>n) return;
    int f=y*(n+1)+x;
    if(x>0 && x<n && y<n) u[f]-=p[y*n+x]-p[y*n+x-1];
    if(y>0 && y<n && x<n) v[f]-=p[y*n+x]-p[(y-1)*n+x];
}
__global__ void advectDye(const float4* dye,float4* out,const float* u,const float* v,int n,float dt,float decay) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=n || y>=n) return;
    float2 a=velocity(u,v,x+0.5f,y+0.5f,n);
    float4 d=sample(dye,x-dt*a.x,y-dt*a.y,n,n,n);
    out[y*n+x]=make_float4(d.x*decay,d.y*decay,d.z*decay,0);
}
__global__ void pixelsKernel(const float4* dye,float4* out,int n) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=n || y>=n) return;
    float4 d=dye[y*n+x];
    out[y*n+x]=make_float4(d.x,d.y,d.z,1);
}
void launched() { checkCuda(cudaGetLastError(), "launch fluid kernel"); }
}
FluidSimulation::FluidSimulation(int n) : size_(n),
    u_(faceCount(n)),uNext_(faceCount(n)),v_(faceCount(n)),vNext_(faceCount(n)),
    dye_(cellCount(n)),dyeNext_(cellCount(n)),pressure_(cellCount(n)),
    pressureNext_(cellCount(n)),divergence_(cellCount(n)) { reset(); }
void FluidSimulation::reset() {
    u_.clear(); uNext_.clear(); v_.clear(); vNext_.clear();
    dye_.clear(); dyeNext_.clear(); pressure_.clear(); pressureNext_.clear(); divergence_.clear();
}
void FluidSimulation::addSplat(const Splat& s) {
    if(!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.radius) || s.radius<=0 ||
       !std::isfinite(s.velocityX) || !std::isfinite(s.velocityY) ||
       !std::isfinite(s.dyeAmount) || s.dyeAmount<0 ||
       !std::isfinite(s.color.x) || !std::isfinite(s.color.y) || !std::isfinite(s.color.z) ||
       s.color.x<0 || s.color.y<0 || s.color.z<0)
        throw std::invalid_argument("Invalid fluid splat");
    splatKernel<<<blocks(size_+1),threads>>>(u_.data(),v_.data(),dye_.data(),size_,s); launched();
}
void FluidSimulation::project() {
    if(settings.pressureIterations<1 || settings.pressureIterations>2000)
        throw std::invalid_argument("Pressure iterations must be in [1, 2000]");
    divergenceKernel<<<blocks(size_),threads>>>(u_.data(),v_.data(),divergence_.data(),size_); launched();
    pressure_.clear(); // deterministic cold start; pressure is defined up to a constant
    for(int k=0;k<settings.pressureIterations;++k) {
        pressureKernel<<<blocks(size_),threads>>>(pressure_.data(),pressureNext_.data(),divergence_.data(),size_);
        launched(); pressure_.swap(pressureNext_);
    }
    gradientKernel<<<blocks(size_+1),threads>>>(u_.data(),v_.data(),pressure_.data(),size_); launched();
}
void FluidSimulation::step(float dt) {
    if(!std::isfinite(dt) || dt<=0 || dt>1.0f/30.0f ||
       !std::isfinite(settings.velocityDecay) || settings.velocityDecay<0 ||
       !std::isfinite(settings.dyeDecay) || settings.dyeDecay<0)
        throw std::invalid_argument("Invalid simulation timestep or decay");
    if(timing) timing->begin();
    advectVelocity<<<blocks(size_+1),threads>>>(u_.data(),v_.data(),uNext_.data(),vNext_.data(),size_,dt,std::exp(-settings.velocityDecay*dt));
    launched(); u_.swap(uNext_); v_.swap(vNext_);
    if(timing) timing->mark(1);
    project();
    if(timing) timing->mark(2);
    advectDye<<<blocks(size_),threads>>>(dye_.data(),dyeNext_.data(),u_.data(),v_.data(),size_,dt,std::exp(-settings.dyeDecay*dt));
    launched(); dye_.swap(dyeNext_);
    if(timing) timing->mark(3);
}
void FluidSimulation::writePixels(float4* pixels) const {
    pixelsKernel<<<blocks(size_),threads>>>(dye_.data(),pixels,size_); launched();
}
Diagnostics FluidSimulation::diagnostics() const {
    std::vector<float> u(size_t(size_+1)*(size_+1)),v(u.size());
    std::vector<float4> d(size_t(size_)*size_);
    checkCuda(cudaMemcpy(u.data(),u_.data(),u_.bytes(),cudaMemcpyDeviceToHost),"read velocity diagnostics");
    checkCuda(cudaMemcpy(v.data(),v_.data(),v_.bytes(),cudaMemcpyDeviceToHost),"read velocity diagnostics");
    checkCuda(cudaMemcpy(d.data(),dye_.data(),dye_.bytes(),cudaMemcpyDeviceToHost),"read dye diagnostics");
    Diagnostics result;
    for(size_t i=0;i<u.size();++i) result.finite &= std::isfinite(u[i]) && std::isfinite(v[i]);
    for(int y=0;y<size_;++y) for(int x=0;x<size_;++x) {
        int i=y*(size_+1)+x;
        double div=double(u[i+1])-u[i]+double(v[i+size_+1])-v[i];
        result.divergenceRms+=div*div;
        result.maxSpeed=std::max(result.maxSpeed,std::hypot(0.5f*(u[i]+u[i+1]),0.5f*(v[i]+v[i+size_+1])));
        auto c=d[y*size_+x]; result.finite &= std::isfinite(c.x)&&std::isfinite(c.y)&&std::isfinite(c.z);
        result.dyeSum+=double(c.x)+c.y+c.z;
    }
    for(int k=0;k<size_;++k)
        result.wallsClosed &= u[k*(size_+1)]==0 && u[k*(size_+1)+size_]==0 && v[k]==0 && v[size_*(size_+1)+k]==0;
    result.divergenceRms=std::sqrt(result.divergenceRms/(size_*size_));
    return result;
}
