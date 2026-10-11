#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pvz::runtime {

// All words are x86 addresses, even when the controlling DLL is x64.
struct ResidentMailbox {
    std::uint32_t command = 1; // idle=0, hold=1, execute=2, stop=3
    std::uint32_t acknowledgement = 0;
    std::uint32_t result = 0;
    std::uint32_t deadline = 0;
    std::uint32_t saved_stack = 0;
    std::uint32_t code = 0;
};

struct ResidentImports {
    std::uint32_t sleep = 0, clock = 0, peek = 0, translate = 0, dispatch = 0;
    std::uint32_t begin_paint = 0, end_paint = 0;
    std::uint32_t wait = 0;
};

// A tiny assembler with only near jumps, so loop size cannot overflow rel8.
class X86Code {
public:
    std::vector<std::uint8_t> bytes;
    void emit(std::initializer_list<std::uint8_t> values) { bytes.insert(bytes.end(), values); }
    void word(std::uint32_t value) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
        bytes.insert(bytes.end(), p, p + 4);
    }
    void push(std::uint32_t value) { emit({0x68}); word(value); }
    void call(std::uint32_t address) { emit({0xb8}); word(address); emit({0xff, 0xd0}); }
    void store(std::uint32_t address, std::uint32_t value) { emit({0xc7, 0x05}); word(address); word(value); }
    std::size_t jump(std::uint8_t condition = 0) {
        if (condition) emit({0x0f, condition}); else emit({0xe9});
        const auto pos = bytes.size(); word(0); return pos;
    }
    void target(std::size_t pos, std::size_t dest) {
        const auto delta = static_cast<std::int32_t>(dest) - static_cast<std::int32_t>(pos + 4);
        std::memcpy(bytes.data() + pos, &delta, 4);
    }
};

inline std::vector<std::uint8_t> build_resident_loop(
    std::uint32_t data, std::uint32_t message, std::uint32_t paint,
    std::uint32_t return_ip, ResidentImports api, bool owner = true, std::uint32_t wake = 0) {
    X86Code a;
    auto wait_command = [&] {
        if (api.wait && wake) {
            a.push(50); a.push(wake); a.call(api.wait);
        } else { a.push(1); a.call(api.sleep); }
    };
    a.emit({0x9c, 0x60}); // preserve original flags and every register
    const auto loop = a.bytes.size();
    a.emit({0x83, 0x3d}); a.word(data); a.emit({0x03});
    const auto stop = a.jump(0x84);
    a.call(api.clock);
    a.emit({0x2b, 0x05}); a.word(data + 12); // signed modular deadline comparison
    a.emit({0x85, 0xc0}); const auto expired = a.jump(0x89);
    a.emit({0xa1}); a.word(data); a.emit({0x83, 0xf8, 0x02});
    const auto execute = a.jump(0x84);
    a.emit({0x83, 0xf8, 0x01}); const auto held = a.jump(0x84);
    a.store(data + 4, 0);
    std::size_t no_message = 0, painted = 0, timer = 0, input_skipped = 0;
    if (owner) {
        a.push(1); a.push(0); a.push(0); a.push(0); a.push(message); a.call(api.peek);
        a.emit({0x85, 0xc0}); no_message = a.jump(0x84);
        a.emit({0x81, 0x3d}); a.word(message + 4); a.word(0x0f);
        const auto paint_jump = a.jump(0x84);
        // Timer handlers must not sneak an unrequested game update into idle.
        a.emit({0x81, 0x3d}); a.word(message + 4); a.word(0x113); timer = a.jump(0x84);
        a.emit({0x83, 0x3d}); a.word(data); a.emit({0x04}); const auto accept_input = a.jump(0x84);
        a.emit({0xf7, 0x05}); a.word(message + 12); a.word(0x10000000);
        const auto lifecycle_input = a.jump(0x85);
        a.emit({0xa1}); a.word(message + 4); a.emit({0x3d}); a.word(0x100);
        const auto below_input = a.jump(0x82);
        a.emit({0x3d}); a.word(0x20e); input_skipped = a.jump(0x86);
        a.target(accept_input, a.bytes.size()); a.target(below_input, a.bytes.size());
        a.target(lifecycle_input, a.bytes.size());
        a.push(message); a.call(api.translate); a.push(message); a.call(api.dispatch);
        painted = a.jump();
        a.target(paint_jump, a.bytes.size());
        if (api.begin_paint && api.end_paint) {
            a.push(paint); a.emit({0xff, 0x35}); a.word(message); a.call(api.begin_paint);
            a.push(paint); a.emit({0xff, 0x35}); a.word(message); a.call(api.end_paint);
        }
    }
    const auto idle = a.bytes.size();
    a.push(1); a.call(api.sleep); auto again = a.jump(); a.target(again, loop);
    if (owner) { a.target(no_message, idle); a.target(painted, loop); a.target(timer, loop); a.target(input_skipped, loop); }
    a.target(held, a.bytes.size()); a.store(data + 4, 1);
    a.call(api.clock); a.emit({0x89, 0xc3}); // EBX = start of held interval
    const auto held_poll = a.bytes.size();
    a.emit({0x83, 0x3d}); a.word(data); a.emit({0x01});
    auto released = a.jump(0x85); a.target(released, loop);
    a.call(api.clock); a.emit({0x89, 0xc2, 0x2b, 0x05}); a.word(data + 12);
    a.emit({0x85, 0xc0}); const auto held_expired = a.jump(0x89);
    a.emit({0x89, 0xd0, 0x2b, 0xc3, 0x83, 0xf8, 0x02});
    const auto idle_hold = a.jump(0x83);
    // Fast RPCs yield briefly; an idle learner never burns a whole CPU core.
    a.push(0); a.call(api.sleep); again = a.jump(); a.target(again, held_poll);
    a.target(idle_hold, a.bytes.size()); wait_command();
    again = a.jump(); a.target(again, held_poll);
    a.target(execute, a.bytes.size());
    a.emit({0x89, 0x25}); a.word(data + 16);
    a.push(0); a.emit({0xff, 0x15}); a.word(data + 20);
    a.emit({0x8b, 0x25}); a.word(data + 16); // supports ret and ret 4 blobs
    a.emit({0xa3}); a.word(data + 8); a.store(data + 4, 2);
    const auto completed = a.bytes.size();
    a.emit({0x83, 0x3d}); a.word(data); a.emit({0x02});
    const auto changed = a.jump(0x85);
    a.call(api.clock); a.emit({0x2b, 0x05}); a.word(data + 12);
    a.emit({0x85, 0xc0}); const auto abandoned = a.jump(0x89);
    a.push(0); a.call(api.sleep); again = a.jump(); a.target(again, completed);
    a.target(changed, loop);
    const auto exit = a.bytes.size(); a.target(stop, exit); a.target(expired, exit); a.target(abandoned, exit); a.target(held_expired, exit);
    a.store(data + 4, 3); a.emit({0x61, 0x9d});
    if (owner) { a.emit({0x68}); a.word(return_ip); a.emit({0xc3}); }
    else a.emit({0x31, 0xc0, 0xc2, 0x04, 0x00});
    return a.bytes;
}
} // namespace pvz::runtime
