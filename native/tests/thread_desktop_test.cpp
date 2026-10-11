#include "../src/process/thread_desktop.h"
#include <cstdio>
#include <cstdlib>
#include <thread>

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s (error %lu)\n", message, GetLastError()); std::exit(1); }
}

static DWORD handles() {
    DWORD count = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &count) != 0, "handle count");
    return count;
}

int main() {
    std::thread normal([] {
        require(pvz::process::ensure_default_desktop(), "bind default");
        DWORD before = handles();
        for (int i = 0; i < 10000; ++i)
            require(pvz::process::ensure_default_desktop(), "repeat default");
        require(handles() == before, "default polling leaked handles");
    });
    normal.join();

    wchar_t name[80];
    swprintf_s(name, L"PvzDesktopTest_%lu", GetCurrentProcessId());
    HDESK alternate = CreateDesktopW(name, nullptr, nullptr, 0, GENERIC_ALL, nullptr);
    require(alternate != nullptr, "create alternate desktop");
    std::thread switched([&] {
        require(SetThreadDesktop(alternate) != 0, "bind alternate desktop");
        DWORD before = handles();
        {
            pvz::process::ThreadDesktop binding;
            require(binding.ensure_default(), "switch to default");
            DWORD bound = handles();
            require(bound <= before + 1, "multiple owned desktop handles");
            for (int i = 0; i < 10000; ++i)
                require(binding.ensure_default(), "repeat switched binding");
            require(handles() == bound, "switched polling leaked handles");
        }
        require(GetThreadDesktop(GetCurrentThreadId()) == alternate, "restore prior desktop");
        require(handles() == before, "owned desktop was not closed");
    });
    switched.join();
    require(CloseDesktop(alternate) != 0, "close alternate desktop");
    std::puts("20000 desktop checks: stable handles, restored binding");
}
