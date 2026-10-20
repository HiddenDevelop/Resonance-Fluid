#pragma once
#include <array>
#include <complex>
#include <cstdint>
#include <span>
#include <vector>

struct StereoSample { float left=0, right=0; };
enum class SampleEncoding { PcmUnsigned8, Pcm16, Pcm24, Pcm32, Float32, Float64 };
struct SampleFormat { SampleEncoding encoding; int channels; int blockAlign; };
// Pure CPU conversion; packed little-endian PCM, including left-aligned 24-in-32.
std::vector<StereoSample> convertSamples(const void* bytes, size_t frames, SampleFormat format, bool silent=false);

struct AudioFeatures {
    float bass=0, mid=0, treble=0; // normalized, attack/release smoothed [0,1]
    float rms=0, loudness=0, onset=0;
    std::array<float,3> rawBands{}; // linear band RMS for diagnostics/tests
    uint64_t onsetId=0, sequence=0;
    double sampleTime=0; // QPC seconds at the center of the analyzed window
    double analysisMs=0;
    bool timestampValid=false;
};

class AudioAnalyzer {
public:
    static constexpr size_t fftSize=2048, hopSize=512;
    explicit AudioAnalyzer(double sampleRate=48000, float smoothing=1);
    // firstSampleTime uses the same QPC clock as the renderer. No Windows dependency here.
    bool push(std::span<const StereoSample> samples, double firstSampleTime, bool timestampValid=true);
    const AudioFeatures& features() const { return features_; }
    void reset();
private:
    void analyze(double time, bool valid);
    double rate_;
    float smoothing_;
    size_t write_=0, filled_=0, sinceHop_=0;
    std::array<StereoSample,fftSize> ring_{};
    std::array<float,fftSize> window_{};
    std::array<float,fftSize/2+1> previousSpectrum_{};
    std::array<float,3> levels_{0.02f,0.02f,0.02f};
    float slowFlux_=0, cooldown_=0, windowEnergy_=0;
    AudioFeatures features_;
};
float smoothEnvelope(float current, float target, float dt, float attack, float release);
