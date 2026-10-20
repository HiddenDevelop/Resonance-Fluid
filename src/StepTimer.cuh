#pragma once
#include "CudaUtils.cuh"
#include <array>
#include <cstdint>

// Optional sampled stream timing. Never waits for the device; busy slots are skipped.
class StepTimer {
    struct Slot { std::array<cudaEvent_t,4> events{}; bool pending=false; };
    std::array<Slot,8> slots_{};
    Slot* active_=nullptr;
    unsigned tick_=0;
public:
    uint64_t samples=0, skipped=0;
    double stepMs=0, velocityMs=0, pressureMs=0, dyeMs=0;
    StepTimer() {
        try { for(auto& s:slots_) for(auto& e:s.events) checkCuda(cudaEventCreate(&e),"create timing event"); }
        catch(...) { release(); throw; }
    }
    ~StepTimer() { release(); }
    StepTimer(const StepTimer&)=delete;
    StepTimer& operator=(const StepTimer&)=delete;
    void release() noexcept {for(auto& s:slots_) for(auto e:s.events) if(e) cudaEventDestroy(e);}
    void collect() {
        for(auto& s:slots_) if(s.pending) {
            auto ready=cudaEventQuery(s.events[3]);
            if(ready==cudaErrorNotReady) continue;
            checkCuda(ready,"query GPU timer");
            float ms[3]{};
            for(int i=0;i<3;++i) checkCuda(cudaEventElapsedTime(&ms[i],s.events[i],s.events[i+1]),"read GPU timer");
            velocityMs+=ms[0]; pressureMs+=ms[1]; dyeMs+=ms[2]; stepMs+=ms[0]+ms[1]+ms[2];
            ++samples; s.pending=false;
        }
    }
    void begin() {
        active_=nullptr;
        if(tick_++%16) return;
        collect();
        for(auto& s:slots_) if(!s.pending) {active_=&s; break;}
        if(!active_) {++skipped; return;}
        checkCuda(cudaEventRecord(active_->events[0]),"start GPU timer");
    }
    void mark(int index) {
        if(!active_) return;
        checkCuda(cudaEventRecord(active_->events[index]),"mark GPU timer");
        if(index==3) {active_->pending=true; active_=nullptr;}
    }
};
