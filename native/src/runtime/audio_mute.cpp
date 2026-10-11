#include "audio_mute.h"
#include <mmdeviceapi.h>
#include <audiopolicy.h>

namespace pvz::runtime {

AudioMute& AudioMute::instance() { static AudioMute value; return value; }

bool AudioMute::set_process_mute_nolock(DWORD pid, bool mute) noexcept {
    if (pid == 0) return false;
    const auto hr_init = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool need_uninit = (hr_init == S_OK || hr_init == S_FALSE);

    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDevice* device = nullptr;
    IAudioSessionManager2* manager = nullptr;
    IAudioSessionEnumerator* sessions = nullptr;
    bool found = false;

    if (SUCCEEDED(::CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator))) &&
        SUCCEEDED(enumerator->GetDefaultAudioEndpoint(eRender, eMultimedia, &device)) &&
        SUCCEEDED(device->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr,
            reinterpret_cast<void**>(&manager))) &&
        SUCCEEDED(manager->GetSessionEnumerator(&sessions))) {

        int count = 0;
        if (SUCCEEDED(sessions->GetCount(&count))) {
            for (int i = 0; i < count; ++i) {
                IAudioSessionControl* control = nullptr;
                if (FAILED(sessions->GetSession(i, &control)) || !control) continue;

                IAudioSessionControl2* owner = nullptr;
                if (SUCCEEDED(control->QueryInterface(__uuidof(IAudioSessionControl2), reinterpret_cast<void**>(&owner))) && owner) {
                    DWORD session_pid = 0;
                    if (SUCCEEDED(owner->GetProcessId(&session_pid)) && session_pid == pid) {
                        ISimpleAudioVolume* volume = nullptr;
                        if (SUCCEEDED(control->QueryInterface(__uuidof(ISimpleAudioVolume), reinterpret_cast<void**>(&volume))) && volume) {
                            if (SUCCEEDED(volume->SetMute(mute ? TRUE : FALSE, nullptr))) {
                                found = true;
                            }
                            volume->Release();
                        }
                    }
                    owner->Release();
                }
                control->Release();
            }
        }
    }

    if (sessions) sessions->Release();
    if (manager) manager->Release();
    if (device) device->Release();
    if (enumerator) enumerator->Release();
    if (need_uninit) ::CoUninitialize();
    return found;
}

void AudioMute::begin(DWORD pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_turbo_pid_ = pid;
    set_process_mute_nolock(pid, true);
}

bool AudioMute::mute_process(DWORD pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    return set_process_mute_nolock(pid, true);
}

bool AudioMute::unmute_process(DWORD pid) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return set_process_mute_nolock(pid, false);
}

void AudioMute::end() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_turbo_pid_) {
        set_process_mute_nolock(current_turbo_pid_, false);
        current_turbo_pid_ = 0;
    }
}

} // namespace pvz::runtime
