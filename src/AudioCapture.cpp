#include "AudioCapture.h"
#include <Windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <ksmedia.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

using Microsoft::WRL::ComPtr;
namespace {
void checked(HRESULT hr,const char* operation) {
    if(FAILED(hr)) {
        std::ostringstream message;message<<operation<<" failed (HRESULT 0x"<<std::hex<<unsigned(hr)<<"). Retrying; visualizer remains available.";
        throw std::runtime_error(message.str());
    }
}
struct ComApartment {
    ComApartment() {checked(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"Initialize audio COM");}
    ~ComApartment() {CoUninitialize();}
};
struct TaskFree {void operator()(void* p) const {CoTaskMemFree(p);}};
std::wstring endpointId(IMMDevice* device) {
    LPWSTR value=nullptr;checked(device->GetId(&value),"Get playback device ID");
    std::unique_ptr<wchar_t,TaskFree> owner(value);return value;
}
SampleFormat describe(const WAVEFORMATEX* f) {
    WORD tag=f->wFormatTag;
    if(tag==WAVE_FORMAT_EXTENSIBLE) {
        if(f->cbSize<sizeof(WAVEFORMATEXTENSIBLE)-sizeof(WAVEFORMATEX)) throw std::runtime_error("Truncated extensible audio format");
        const auto* ex=reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(f);
        if(IsEqualGUID(ex->SubFormat,KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)) tag=WAVE_FORMAT_IEEE_FLOAT;
        else if(IsEqualGUID(ex->SubFormat,KSDATAFORMAT_SUBTYPE_PCM)) tag=WAVE_FORMAT_PCM;
        else throw std::runtime_error("Unsupported Windows audio subformat; select PCM playback format");
        if(ex->Samples.wValidBitsPerSample>f->wBitsPerSample) throw std::runtime_error("Invalid PCM valid-bit count");
    }
    SampleEncoding encoding;
    if(tag==WAVE_FORMAT_IEEE_FLOAT && f->wBitsPerSample==32) encoding=SampleEncoding::Float32;
    else if(tag==WAVE_FORMAT_IEEE_FLOAT && f->wBitsPerSample==64) encoding=SampleEncoding::Float64;
    else if(tag==WAVE_FORMAT_PCM) {
        switch(f->wBitsPerSample) {
        case 8:encoding=SampleEncoding::PcmUnsigned8;break;
        case 16:encoding=SampleEncoding::Pcm16;break;
        case 24:encoding=SampleEncoding::Pcm24;break;
        case 32:encoding=SampleEncoding::Pcm32;break;
        default:throw std::runtime_error("Unsupported PCM width");
        }
    } else throw std::runtime_error("Unsupported Windows mix format");
    return {encoding,int(f->nChannels),int(f->nBlockAlign)};
}
void sleepUntilStop(std::stop_token stop,int milliseconds) {
    std::mutex mutex;std::condition_variable_any cv;std::unique_lock lock(mutex);
    cv.wait_for(lock,stop,std::chrono::milliseconds(milliseconds),[]{return false;});
}
}
double audioClockSeconds() {
    static const double frequency=[] {LARGE_INTEGER f;QueryPerformanceFrequency(&f);return double(f.QuadPart);}();
    LARGE_INTEGER now;QueryPerformanceCounter(&now);return double(now.QuadPart)/frequency;
}
AudioCapture::AudioCapture(std::wstring device,float smoothing):worker_([this,device=std::move(device),smoothing](std::stop_token stop){run(stop,device,smoothing);}) {}
AudioCapture::~AudioCapture() {worker_.request_stop();worker_.join();}
bool AudioCapture::tryRead(AudioSnapshot& result) {
    std::unique_lock lock(mutex_,std::try_to_lock);if(!lock.owns_lock()) return false;
    result=latest_;return true;
}
void AudioCapture::publish(const AudioSnapshot& value) {std::lock_guard lock(mutex_);latest_=value;}
void AudioCapture::run(std::stop_token stop,std::wstring requested,float smoothing) {
    AudioSnapshot snapshot;
    try {
        ComApartment apartment;
        while(!stop.stop_requested()) {
            try {
                ComPtr<IMMDeviceEnumerator> enumerator;
                checked(CoCreateInstance(__uuidof(MMDeviceEnumerator),nullptr,CLSCTX_ALL,IID_PPV_ARGS(&enumerator)),"Create playback enumerator");
                ComPtr<IMMDevice> device;
                if(requested.empty()) checked(enumerator->GetDefaultAudioEndpoint(eRender,eMultimedia,&device),"Open default playback device");
                else checked(enumerator->GetDevice(requested.c_str(),&device),"Open selected playback device");
                const auto id=endpointId(device.Get());
                ComPtr<IAudioClient> client;checked(device->Activate(__uuidof(IAudioClient),CLSCTX_ALL,nullptr,reinterpret_cast<void**>(client.GetAddressOf())),"Activate WASAPI");
                WAVEFORMATEX* mix=nullptr;checked(client->GetMixFormat(&mix),"Get playback mix format");
                std::unique_ptr<WAVEFORMATEX,TaskFree> formatOwner(mix);
                auto format=describe(mix);
                AudioAnalyzer analyzer(mix->nSamplesPerSec,smoothing);
                checked(client->Initialize(AUDCLNT_SHAREMODE_SHARED,AUDCLNT_STREAMFLAGS_LOOPBACK,1000000,0,mix,nullptr),"Initialize WASAPI loopback");
                ComPtr<IAudioCaptureClient> capture;checked(client->GetService(IID_PPV_ARGS(&capture)),"Get loopback service");
                checked(client->Start(),"Start loopback");
                struct StopClient {IAudioClient* p;~StopClient(){p->Stop();}} stopClient{client.Get()};
                ++snapshot.generation;snapshot.features={};snapshot.capturing=true;
                snapshot.status="WASAPI loopback: "+std::to_string(mix->nSamplesPerSec)+" Hz, "+std::to_string(mix->nChannels)+" channels";
                snapshot.publishedTime=audioClockSeconds();publish(snapshot);
                double lastPacket=audioClockSeconds(),lastSilence=lastPacket,lastDeviceCheck=lastPacket;
                bool gap=false;
                std::vector<StereoSample> silence(std::max<DWORD>(1,mix->nSamplesPerSec/100));
                while(!stop.stop_requested()) {
                    UINT32 available=0;checked(capture->GetNextPacketSize(&available),"Get loopback packet size");
                    while(available && !stop.stop_requested()) {
                        BYTE* data=nullptr;UINT32 frames=0;DWORD flags=0;UINT64 position=0,qpc=0;
                        checked(capture->GetBuffer(&data,&frames,&flags,&position,&qpc),"Read loopback packet");
                        std::vector<StereoSample> samples;
                        try {samples=convertSamples(data,frames,format,(flags&AUDCLNT_BUFFERFLAGS_SILENT)!=0);}
                        catch(...) {capture->ReleaseBuffer(frames);throw;}
                        checked(capture->ReleaseBuffer(frames),"Release loopback packet");
                        if(gap || (flags&AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY)) analyzer.reset();
                        gap=false;lastPacket=audioClockSeconds();++snapshot.packets;
                        bool valid=(flags&AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR)==0 && qpc!=0;
                        double timestamp=valid?double(qpc)*1e-7:lastPacket-double(frames)/mix->nSamplesPerSec;
                        if(analyzer.push(samples,timestamp,valid)) {
                            snapshot.features=analyzer.features();snapshot.publishedTime=audioClockSeconds();publish(snapshot);
                        }
                        checked(capture->GetNextPacketSize(&available),"Drain loopback packets");
                    }
                    double now=audioClockSeconds();
                    // Some endpoints deliver no packets at all during silence. Drive release envelopes with zeros.
                    if(now-lastPacket>0.1 && now-lastSilence>=0.01) {
                        gap=true;lastSilence=now;
                        if(analyzer.push(silence,now-double(silence.size())/mix->nSamplesPerSec,false)) {
                            snapshot.features=analyzer.features();snapshot.publishedTime=now;publish(snapshot);
                        }
                    }
                    // Polling avoids COM notification callbacks/lifetime complexity; reconnection is <= ~1 s.
                    if(requested.empty() && now-lastDeviceCheck>=1) {
                        lastDeviceCheck=now;ComPtr<IMMDevice> current;
                        checked(enumerator->GetDefaultAudioEndpoint(eRender,eMultimedia,&current),"Check default playback device");
                        if(endpointId(current.Get())!=id) break;
                    }
                    sleepUntilStop(stop,5);
                }
            } catch(const std::exception& error) {
                snapshot.capturing=false;snapshot.features={};snapshot.status=error.what();snapshot.publishedTime=audioClockSeconds();publish(snapshot);
                sleepUntilStop(stop,1000);
            }
        }
    } catch(const std::exception& error) {
        snapshot.capturing=false;snapshot.features={};snapshot.status=error.what();publish(snapshot);
    }
}
void AudioCapture::listDevices() {
    ComApartment apartment;ComPtr<IMMDeviceEnumerator> enumerator;
    checked(CoCreateInstance(__uuidof(MMDeviceEnumerator),nullptr,CLSCTX_ALL,IID_PPV_ARGS(&enumerator)),"Create device enumerator");
    ComPtr<IMMDeviceCollection> collection;checked(enumerator->EnumAudioEndpoints(eRender,DEVICE_STATE_ACTIVE,&collection),"List playback endpoints");
    UINT count=0;checked(collection->GetCount(&count),"Count endpoints");
    for(UINT i=0;i<count;++i) {
        ComPtr<IMMDevice> device;checked(collection->Item(i,&device),"Read endpoint");
        ComPtr<IPropertyStore> properties;checked(device->OpenPropertyStore(STGM_READ,&properties),"Read device properties");
        PROPVARIANT name;PropVariantInit(&name);
        HRESULT hr=properties->GetValue(PKEY_Device_FriendlyName,&name);
        if(SUCCEEDED(hr) && name.vt==VT_LPWSTR) std::wcout<<name.pwszVal<<L'\n';
        PropVariantClear(&name);std::wcout<<L"  "<<endpointId(device.Get())<<L'\n';
    }
}

