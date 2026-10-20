#include "AudioAnalysis.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace {
int sampleBytes(SampleEncoding e) {
    switch(e) {
    case SampleEncoding::PcmUnsigned8:return 1;
    case SampleEncoding::Pcm16:return 2;
    case SampleEncoding::Pcm24:return 3;
    case SampleEncoding::Pcm32:case SampleEncoding::Float32:return 4;
    case SampleEncoding::Float64:return 8;
    } throw std::invalid_argument("Unknown audio encoding");
}
float decode(const unsigned char* p,SampleEncoding e) {
    double value=0;
    switch(e) {
    case SampleEncoding::PcmUnsigned8: value=(int(*p)-128)/128.0; break;
    case SampleEncoding::Pcm16: {int16_t v;std::memcpy(&v,p,2);value=v/32768.0;break;}
    case SampleEncoding::Pcm24: {int32_t v=p[0]|(int32_t(p[1])<<8)|(int32_t(p[2])<<16);if(v&0x800000) v-=0x1000000;value=v/8388608.0;break;}
    case SampleEncoding::Pcm32: {int32_t v;std::memcpy(&v,p,4);value=v/2147483648.0;break;}
    case SampleEncoding::Float32: {float v;std::memcpy(&v,p,4);value=v;break;}
    case SampleEncoding::Float64: {double v;std::memcpy(&v,p,8);value=v;break;}
    }
    return std::isfinite(value)?float(std::clamp(value,-1.0,1.0)):0.0f;
}
// In-place radix-2 FFT. Small enough that a library/FFT plan adds little value here.
void fft(std::array<std::complex<float>,AudioAnalyzer::fftSize>& a) {
    constexpr size_t n=AudioAnalyzer::fftSize;
    for(size_t i=1,j=0;i<n;++i) {
        size_t bit=n>>1;for(;j&bit;bit>>=1) j^=bit;j^=bit;
        if(i<j) std::swap(a[i],a[j]);
    }
    for(size_t length=2;length<=n;length*=2) {
        const auto rotation=std::polar(1.0f,-2*std::numbers::pi_v<float>/float(length));
        for(size_t start=0;start<n;start+=length) {
            std::complex<float> w(1,0);
            for(size_t j=0;j<length/2;++j) {
                auto even=a[start+j],odd=w*a[start+j+length/2];
                a[start+j]=even+odd;a[start+j+length/2]=even-odd;w*=rotation;
            }
        }
    }
}
}
std::vector<StereoSample> convertSamples(const void* bytes,size_t frames,SampleFormat f,bool silent) {
    int size=sampleBytes(f.encoding);
    if(f.channels<1 || f.channels>32 || f.blockAlign<f.channels*size || (!bytes && !silent && frames))
        throw std::invalid_argument("Invalid audio sample layout");
    std::vector<StereoSample> output(frames);
    if(silent) return output; // WASAPI permits null data for silent packets
    auto data=static_cast<const unsigned char*>(bytes);
    for(size_t i=0;i<frames;++i) {
        auto p=data+i*f.blockAlign;
        float left=0,right=0;int nl=0,nr=0;
        for(int c=0;c<f.channels;++c) {
            float value=decode(p+c*size,f.encoding);
            if(c%2) {right+=value;++nr;} else {left+=value;++nl;}
        }
        output[i]={left/nl,nr?right/nr:left/nl};
    }
    return output;
}
float smoothEnvelope(float current,float target,float dt,float attack,float release) {
    return target+(current-target)*std::exp(-dt/(target>current?attack:release));
}
AudioAnalyzer::AudioAnalyzer(double rate,float smoothing):rate_(rate),smoothing_(smoothing) {
    if(!std::isfinite(rate) || rate<8000 || rate>192000 || !std::isfinite(smoothing) || smoothing<0.25f || smoothing>4)
        throw std::invalid_argument("Unsupported analysis rate or smoothing");
    for(size_t i=0;i<fftSize;++i) {
        window_[i]=0.5f-0.5f*std::cos(2*std::numbers::pi_v<float>*float(i)/float(fftSize-1));
        windowEnergy_+=window_[i]*window_[i];
    }
}
void AudioAnalyzer::reset() {
    write_=filled_=sinceHop_=0;ring_.fill({});previousSpectrum_.fill(0);levels_.fill(0.02f);
    slowFlux_=cooldown_=0;
    // Event/sequence IDs stay monotonic across discontinuities within this analyzer.
    auto event=features_.onsetId,sequence=features_.sequence;features_={};features_.onsetId=event;features_.sequence=sequence;
}
bool AudioAnalyzer::push(std::span<const StereoSample> samples,double time,bool valid) {
    bool changed=false;
    for(size_t i=0;i<samples.size();++i) {
        auto s=samples[i];
        if(!std::isfinite(s.left)) s.left=0;if(!std::isfinite(s.right)) s.right=0;
        ring_[write_]=s;write_=(write_+1)%fftSize;filled_=std::min(filled_+1,fftSize);++sinceHop_;
        if(filled_==fftSize && sinceHop_>=hopSize) {
            sinceHop_=0;
            analyze(time+(double(i)-(fftSize-1)*0.5)/rate_,valid);changed=true;
        }
    }
    return changed;
}
void AudioAnalyzer::analyze(double time,bool valid) {
    auto start=std::chrono::steady_clock::now();
    std::array<float,fftSize/2+1> power{};
    double squareSum=0;
    for(int channel=0;channel<2;++channel) {
        std::array<std::complex<float>,fftSize> spectrum;
        double mean=0;
        for(auto s:ring_) mean+=channel?s.right:s.left;
        mean/=fftSize;
        for(size_t i=0;i<fftSize;++i) {
            auto sample=ring_[(write_+i)%fftSize];float value=channel?sample.right:sample.left;
            squareSum+=double(value)*value;
            spectrum[i]=float(value-mean)*window_[i];
        }
        fft(spectrum);
        for(size_t k=1;k<fftSize/2;++k) power[k]+=std::norm(spectrum[k])/(fftSize*windowEnergy_);
    }
    // Power combines two channels / 2 and the one-sided spectrum factor of 2.
    float rms=float(std::sqrt(squareSum/(2*fftSize))),flux=0;
    std::array<float,3> bands{};
    for(size_t k=1;k<fftSize/2;++k) {
        double hz=k*rate_/fftSize;
        if(hz>=25 && hz<12000) bands[hz<250?0:hz<2000?1:2]+=power[k];
        float amplitude=std::sqrt(power[k]);flux+=std::max(0.0f,amplitude-previousSpectrum_[k]);
        previousSpectrum_[k]=amplitude;
    }
    const float dt=float(hopSize/rate_);
    cooldown_=std::max(0.0f,cooldown_-dt);
    features_.onset*=std::exp(-dt/0.12f);
    if(rms>0.002f && flux>std::max(0.015f,slowFlux_*2.5f) && cooldown_==0) {
        features_.onset=std::clamp(flux/(0.02f+slowFlux_*3),0.0f,1.0f);
        ++features_.onsetId;cooldown_=0.18f;
    }
    slowFlux_=smoothEnvelope(slowFlux_,flux,dt,0.15f,1.0f);
    float* smoothed[]={&features_.bass,&features_.mid,&features_.treble};
    for(int i=0;i<3;++i) {
        float raw=std::sqrt(bands[i]);features_.rawBands[i]=raw;
        levels_[i]=std::max(raw,levels_[i]*std::exp(-dt/3));
        float target=rms<0.0003f?0:std::clamp(raw/(0.004f+levels_[i]*1.1f),0.0f,1.0f);
        *smoothed[i]=smoothEnvelope(*smoothed[i],target,dt,0.035f*smoothing_,0.35f*smoothing_);
    }
    features_.rms=rms;
    features_.loudness=smoothEnvelope(features_.loudness,std::clamp(rms/0.12f,0.0f,1.0f),dt,0.03f*smoothing_,0.4f*smoothing_);
    features_.sampleTime=time;features_.timestampValid=valid;++features_.sequence;
    features_.analysisMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
