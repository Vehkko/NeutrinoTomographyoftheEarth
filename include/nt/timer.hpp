#pragma once

#include <chrono>

namespace nt {

    struct Timer {
        using Clock = std::chrono::steady_clock;

        Clock::time_point start = Clock::now();

        void reset() noexcept { start = Clock::now(); }

        [[nodiscard]] double seconds() const noexcept {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }

        [[nodiscard]] double milliseconds() const noexcept {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
    };

} // namespace nt
