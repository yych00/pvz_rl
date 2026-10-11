#pragma once

#include <cstdlib>
#include <iostream>

// CTest builds in Release, where standard assert() is compiled out.
#define CHECK(expr) do { if (!(expr)) { \
    std::cerr << __FILE__ << ':' << __LINE__ << ": " << #expr << '\n'; \
    std::abort(); \
} } while (false)
