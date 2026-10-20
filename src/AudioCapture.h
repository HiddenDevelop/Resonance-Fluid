#pragma once
#include "AudioAnalysis.h"
#include <mutex>
#include <string>
#include <thread>

// QPC seconds: same epoch as WASAPI's QPC timestamps after conversion from 100 ns.
double audioClockSeconds();
struct AudioSnapshot {
    AudioFeatures features;
    uint64_t generation=0, packets=0;
    double publishedTime=0;
    bool capturing=false;
    std::string status="Starting audio capture";
};
class AudioCapture {
public:
    explicit AudioCapture(std::wstring deviceId={},float smoothing=1);
    ~AudioCapture();
    AudioCapture(const AudioCapture&)=delete;
    AudioCapture& operator=(const AudioCapture&)=delete;
    // The render thread never waits on the worker. A missed read keeps the previous snapshot.
    bool tryRead(AudioSnapshot& result);
    static void listDevices();
private:
    void run(std::stop_token stop,std::wstring deviceId,float smoothing);
    void publish(const AudioSnapshot& value);
    std::mutex mutex_;
    AudioSnapshot latest_;
    std::jthread worker_;
};
