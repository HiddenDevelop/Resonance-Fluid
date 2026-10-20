#include "AudioCapture.h"
#include <chrono>
#include <iostream>
#include <thread>
int main(int argc,char** argv) {
    if(argc>1 && std::string(argv[1])=="--list-devices") {AudioCapture::listDevices();return 0;}
    AudioCapture capture;AudioSnapshot s;uint64_t last=0;double start=audioClockSeconds();
    while(audioClockSeconds()-start<10) {
        if(capture.tryRead(s) && s.features.sequence!=last) {
            last=s.features.sequence;
            std::cout<<s.status<<" packets="<<s.packets<<" rms="<<s.features.rms<<" bass="<<s.features.bass
                <<" mid="<<s.features.mid<<" treble="<<s.features.treble<<" onset="<<s.features.onsetId<<std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return s.capturing?0:1;
}
