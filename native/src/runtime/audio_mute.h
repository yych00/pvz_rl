#pragma once
#include <Windows.h>
#include <mutex>

namespace pvz::runtime {

// 进程级音频静音管理器：通过 Windows WASAPI 控制指定 PID 的静音状态。
// 所有操作使用局部短期 COM 实例并由互斥锁严格保护，不持久保存裸 COM 接口指针，
// 杜绝跨套间与多线程数据竞争导致的 AUDIOSES.DLL 访问违规 (0xc0000005)。
class AudioMute {
public:
    static AudioMute& instance();
    void begin(DWORD pid);
    void end() noexcept;
    bool mute_process(DWORD pid);
    bool unmute_process(DWORD pid) noexcept;
private:
    AudioMute() = default;
    bool set_process_mute_nolock(DWORD pid, bool mute) noexcept;

    std::mutex mutex_;
    DWORD current_turbo_pid_ = 0;
};

} // namespace pvz::runtime
