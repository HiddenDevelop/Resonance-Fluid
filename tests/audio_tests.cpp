#include "AudioAnalysis.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::vector<StereoSample> tone(float hz,double rate,double seconds,float amplitude=0.2f,bool antiphase=false) {
    std::vector<StereoSample> result(size_t(rate*seconds));
    for(size_t i=0;i<result.size();++i) {float s=amplitude*std::sin(float(2*std::numbers::pi*hz*i/rate));result[i]={s,antiphase?-s:s};}
    return result;
}
}
int main() {
    try {
        int16_t pcm16[]={-32768,32767,0,16384};
        auto out=convertSamples(pcm16,2,{SampleEncoding::Pcm16,2,4});
        require(out[0].left==-1 && out[0].right>0.999 && out[1].right==0.5f,"PCM16 stereo");
        unsigned char pcm24[]={0,0,128,255,255,127};
        out=convertSamples(pcm24,2,{SampleEncoding::Pcm24,1,3});
        require(out[0].left==-1 && out[1].right>0.999,"PCM24 sign extension");
        int32_t pcm32[]={INT32_MIN,1073741824};
        out=convertSamples(pcm32,2,{SampleEncoding::Pcm32,1,4});
        require(out[0].left==-1 && out[1].left==0.5f,"PCM32 / left-aligned 24-in-32");
        unsigned char pcm8[]={0,128,255};out=convertSamples(pcm8,3,{SampleEncoding::PcmUnsigned8,1,1});
        require(out[0].left==-1 && out[1].left==0 && out[2].left>0.99,"Unsigned PCM8");
        float floating[]={std::numeric_limits<float>::quiet_NaN(),2,-0.25f,0.5f};
        out=convertSamples(floating,2,{SampleEncoding::Float32,2,8});
        require(out[0].left==0 && out[0].right==1 && out[1].left==-0.25f,"Float32 finite/clamp");
        double doubles[]={-0.125,0.25};out=convertSamples(doubles,2,{SampleEncoding::Float64,1,8});
        require(out[0].right==-0.125f && out[1].left==0.25f,"Float64 mono");
        out=convertSamples(nullptr,10,{SampleEncoding::Float32,2,8},true);
        require(out.size()==10 && out[9].left==0,"Silent null WASAPI packet");
        bool rejected=false;try {convertSamples(nullptr,10,{SampleEncoding::Pcm16,2,1});} catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"Invalid sample layout rejection");
        for(double rate:{44100.0,48000.0,96000.0}) for(int band=0;band<3;++band) {
            AudioAnalyzer analyzer(rate);float hz[]={100,800,6000};
            auto samples=tone(hz[band],rate,1.0,0.2f,true);analyzer.push(samples,10);
            auto f=analyzer.features();
            require(std::abs(f.rms-0.2f/std::sqrt(2.0f))<0.008f,"Stereo RMS including antiphase audio");
            require(f.rawBands[band]>0.12f,"Tone must land in expected band");
            for(int j=0;j<3;++j) if(j!=band) require(f.rawBands[j]<0.015f,"Tone must not leak strongly to other bands");
            require(f.onsetId<=2,"Steady tone must not repeatedly trigger onsets");
            require(f.timestampValid && f.sampleTime>10.9 && f.sampleTime<11,"Window center timestamp");
        }
        AudioAnalyzer silence;std::vector<StereoSample> zeros(48000*3);
        silence.push(zeros,0);auto f=silence.features();
        require(f.rms==0 && f.bass==0 && f.mid==0 && f.treble==0 && f.onsetId==0,"Silence must not amplify noise or create onsets");
        AudioAnalyzer impulse;impulse.push(zeros,0);
        auto pulse=tone(80,48000,0.08,0.7f);impulse.push(pulse,3);
        require(impulse.features().onsetId>0,"Synthetic bass burst must trigger onset");
        auto first=impulse.features().onsetId;
        impulse.push(zeros,3.08);f=impulse.features();
        require(f.bass<0.001f && f.loudness<0.001f && f.onsetId==first,"Release must settle without silence retriggers");
        impulse.push(pulse,6.08);require(impulse.features().onsetId>first,"Separated burst must retrigger");
        float attack=smoothEnvelope(0,1,0.05f,0.02f,0.4f),release=smoothEnvelope(1,0,0.05f,0.02f,0.4f);
        require(attack>0.9f && release>0.85f,"Separate attack/release smoothing");
        auto signal=tone(900,48000,1);
        AudioAnalyzer whole,chunked;whole.push(signal,0);
        for(size_t i=0;i<signal.size();i+=137) chunked.push(std::span(signal).subspan(i,std::min(size_t(137),signal.size()-i)),double(i)/48000);
        require(std::abs(whole.features().mid-chunked.features().mid)<1e-6f,"Packet partition must not affect features");
        AudioAnalyzer fast(48000,0.25f),slow(48000,4);auto shortTone=tone(100,48000,0.08);
        fast.push(shortTone,0);slow.push(shortTone,0);require(fast.features().bass>slow.features().bass,"Smoothing control must affect attack");
        std::cout<<"PASS: conversion, FFT bands, stereo RMS, silence, onsets, smoothing and packet partition tests\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
