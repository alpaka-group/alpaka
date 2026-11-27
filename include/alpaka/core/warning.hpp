/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <string>

namespace alpaka::core
{
    namespace detail
    {
        /** State shared by all warnings, guards deduplication and the global mute switch. */
        struct WarningState
        {
            static WarningState& get()
            {
                static WarningState instance;
                return instance;
            }

            std::mutex guard;
            std::set<std::string> alreadyReported;
            bool muted = std::getenv("ALPAKA_NO_WARNINGS") != nullptr;

        private:
            WarningState() = default;
        };
    } // namespace detail

    /** Report a runtime condition the user should know about but which is not an error.
     *
     * @attention Library internal. This is not part of the alpaka user API, applications must not call it.
     *
     * @details
     * Used where alpaka cannot deliver what was requested but can continue with a documented fallback, for example
     * when a memory property is supported by the device but the underlying library cannot resolve it on the machine
     * the program runs on.
     *
     * Unlike ALPAKA_LOG_INFO this is always compiled in, because a silently dropped request is exactly what the
     * warning is there to prevent. Each distinct message is printed only once per process so that a warning raised
     * inside an allocation does not flood the output. Set the environment variable ALPAKA_NO_WARNINGS to silence
     * all warnings.
     *
     * @param message Warning text without a trailing newline.
     */
    inline void warn(std::string const& message)
    {
        auto& state = detail::WarningState::get();
        if(state.muted)
            return;

        {
            std::lock_guard<std::mutex> const lock{state.guard};
            if(!state.alreadyReported.insert(message).second)
                return;
        }

        std::cerr << "[alpaka][warning] " << message << std::endl;
    }
} // namespace alpaka::core
