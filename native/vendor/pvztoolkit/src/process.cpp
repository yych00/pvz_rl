
#include "../inc/process.h"

namespace Pt
{

Process::Process()
{
    this->hwnd = nullptr;
    this->pid = 0;
    this->handle = nullptr;
}

Process::~Process()
{
    if (this->handle)
        CloseHandle(this->handle);
}

void Process::BindProcess(DWORD process_id)
{
    if (bound_pid == process_id) return;
    if (handle) CloseHandle(handle);
    hwnd = nullptr;
    handle = nullptr;
    pid = 0;
    bound_pid = process_id;
}

bool Process::OpenByWindow(const wchar_t *class_name, const wchar_t *window_name)
{
    if (this->handle)
        CloseHandle(this->handle);

    this->hwnd = nullptr;
    this->pid = 0;
    this->handle = nullptr;

    if (bound_pid) {
        struct Search { DWORD pid; const wchar_t* cls; const wchar_t* title; HWND found; };
        Search search{bound_pid, class_name, window_name, nullptr};
        EnumWindows([](HWND win, LPARAM param) -> BOOL {
            auto& s = *reinterpret_cast<Search*>(param);
            DWORD owner = 0;
            GetWindowThreadProcessId(win, &owner);
            if (owner != s.pid) return TRUE;
            wchar_t text[256]{};
            if (s.cls && (!GetClassNameW(win, text, 256) || std::wstring(text) != s.cls)) return TRUE;
            if (s.title && (!GetWindowTextW(win, text, 256) || std::wstring(text) != s.title)) return TRUE;
            s.found = win;
            return FALSE;
        }, reinterpret_cast<LPARAM>(&search));
        this->hwnd = search.found;
    } else {
        this->hwnd = FindWindowW(class_name, window_name);
    }
    if (this->hwnd != nullptr)
    {
        GetWindowThreadProcessId(this->hwnd, &(this->pid));
        if (this->pid != 0)
        {
            this->handle = OpenProcess(PROCESS_ALL_ACCESS, false, this->pid);
            if (this->handle != nullptr)
            {
                //
            }
        }
    }

    // assert(PROCESS_ALL_ACCESS == 0x001FFFFF);

#ifdef _DEBUG
    std::wcout << L"查找窗口: " << (class_name == nullptr ? L"nullptr" : class_name)               //
               << L" " << (window_name == nullptr ? L"nullptr" : window_name) << std::endl         //
               << L" -> " << this->hwnd << L" " << this->pid << L" " << this->handle << std::endl; //
#endif

    // 返回的是窗口有没有找到而不是进程有没有打开
    return this->hwnd != nullptr;
}

bool Process::IsValid()
{
    if (this->handle == nullptr)
        return false;

    DWORD exit_code;
    BOOL ret = GetExitCodeProcess(this->handle, &exit_code);
    bool valid = (ret != 0 && exit_code == STILL_ACTIVE);

#ifdef _DEBUG
    if (!valid)
        std::wcout << L"当前进程不可用." << std::endl;
#endif

    return valid;
}

} // namespace Pt
