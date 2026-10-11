#include "../src/process/thread_safe_point.h"
#include "test_check.h"

#include <sstream>
#include <string>
#include <vector>

int wmain(int argc, wchar_t** argv) {
    CHECK(argc == 2);
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE output_read, output_write, input_read, input_write;
    CHECK(::CreatePipe(&output_read, &output_write, &security, 0));
    CHECK(::CreatePipe(&input_read, &input_write, &security, 0));
    CHECK(::SetHandleInformation(output_read, HANDLE_FLAG_INHERIT, 0));
    CHECK(::SetHandleInformation(input_write, HANDLE_FLAG_INHERIT, 0));
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = input_read;
    startup.hStdOutput = output_write;
    startup.hStdError = output_write;
    std::wstring command = L"\"" + std::wstring(argv[1]) + L"\" --spin-fixture";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(0);
    PROCESS_INFORMATION child{};
    CHECK(::CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr,
        TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child));
    ::CloseHandle(input_read);
    ::CloseHandle(output_write);

    std::string line;
    char character = 0;
    DWORD bytes = 0;
    while (::ReadFile(output_read, &character, 1, &bytes, nullptr) && bytes == 1) {
        if (character == '\n') break;
        line.push_back(character);
        CHECK(line.size() < 128);
    }
    std::uintptr_t instruction = 0;
    DWORD thread_id = 0;
    std::istringstream fields(line);
    fields >> std::hex >> instruction >> std::dec >> thread_id;
    CHECK(instruction && thread_id);
    HANDLE thread = ::OpenThread(
        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
        FALSE, thread_id);
    CHECK(thread);
    using namespace pvz::process;
    DWORD observed = 0;
    auto status = SafePointStatus::Retry;
    for (int attempt = 0; attempt < 100 && status == SafePointStatus::Retry; ++attempt) {
        status = try_suspend_at_safe_point(thread, instruction, &observed);
        if (status == SafePointStatus::Retry) ::Sleep(1);
    }
    CHECK(status == SafePointStatus::Acquired);
    CHECK(observed == instruction);
    CHECK(::ResumeThread(thread) == 1);
    CHECK(try_suspend_at_safe_point(thread, 1) == SafePointStatus::Retry);
    CHECK(::SuspendThread(thread) == 0);
    CHECK(::ResumeThread(thread) == 1);
    ::CloseHandle(thread);

    CHECK(::WriteFile(input_write, "\n", 1, &bytes, nullptr));
    ::CloseHandle(input_write);
    CHECK(::WaitForSingleObject(child.hProcess, 5000) == WAIT_OBJECT_0);
    DWORD exit_code = 1;
    CHECK(::GetExitCodeProcess(child.hProcess, &exit_code) && exit_code == 0);
    ::CloseHandle(output_read);
    ::CloseHandle(child.hThread);
    ::CloseHandle(child.hProcess);
    std::cout << "x64 host / x86 target: WOW64 EIP validation and retry suspension balance passed\n";
}
