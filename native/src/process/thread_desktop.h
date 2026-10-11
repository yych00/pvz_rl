#pragma once

#include <Windows.h>
#include <cwchar>

namespace pvz::process {

// A thread must release its desktop binding before closing the opened handle.
// Keep one binding for the thread lifetime instead of opening on every poll.
class ThreadDesktop {
public:
    ~ThreadDesktop() {
        if (owned_ && ::SetThreadDesktop(previous_)) ::CloseDesktop(owned_);
    }

    bool ensure_default() {
        if (ready_) return true;
        previous_ = ::GetThreadDesktop(::GetCurrentThreadId());
        wchar_t name[256]{};
        DWORD required = 0;
        if (previous_ && ::GetUserObjectInformationW(previous_, UOI_NAME,
                name, sizeof(name), &required) && std::wcscmp(name, L"Default") == 0) {
            ready_ = true;
            return true;
        }
        HDESK desktop = ::OpenDesktopW(L"Default", 0, FALSE,
            DESKTOP_READOBJECTS | DESKTOP_SWITCHDESKTOP);
        if (!desktop) return false;
        if (!::SetThreadDesktop(desktop)) {
            ::CloseDesktop(desktop);
            return false;
        }
        owned_ = desktop;
        ready_ = true;
        return true;
    }

    ThreadDesktop() = default;
    ThreadDesktop(const ThreadDesktop&) = delete;
    ThreadDesktop& operator=(const ThreadDesktop&) = delete;

private:
    HDESK previous_ = nullptr; // Borrowed from GetThreadDesktop.
    HDESK owned_ = nullptr;
    bool ready_ = false;
};

inline bool ensure_default_desktop() {
    thread_local ThreadDesktop desktop;
    return desktop.ensure_default();
}

} // namespace pvz::process
