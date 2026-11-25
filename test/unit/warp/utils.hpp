/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace alpaka::test::warp
{
    using WarpTestBackends
        = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

    // Marks the shared success flag false when a lane detects a failure without aborting execution.
    template<typename SuccessView>
    constexpr void warpCheck(SuccessView success, bool condition) requires requires { success[0u] = false; }
    {
        if(!condition)
        {
            // Trip the shared success flag without aborting the kernel.
            success[0u] = false;
        }
    }

    /**! Overload to write the status to a boolean
     *
     * @param success Set to false if condition is false else do nothing.
     */
    constexpr void warpCheck(bool& success, bool condition)
    {
        if(!condition)
        {
            // Trip the shared success flag without aborting the kernel.
            success = false;
        }
    }

    // Builds a mask with all lanes active for the provided warp size.
    constexpr std::uint64_t fullMask(std::uint32_t warpSize)
    {
        if(warpSize == 0u)
        {
            return 0u;
        }
        if(warpSize >= 64u)
        {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return (std::uint64_t{1} << warpSize) - 1u;
    }

    // Produces a mask that enables every even-numbered lane up to the warp size.
    constexpr std::uint64_t evenMask(std::uint32_t warpSize)
    {
        auto const limit = warpSize < 64u ? warpSize : 64u;
        std::uint64_t mask = 0u;
        for(std::uint32_t lane = 0u; lane < limit; lane += 2u)
        {
            // Populate only even-numbered lanes to model a checkerboard mask.
            mask |= (std::uint64_t{1} << lane);
        }
        return mask;
    }

    // Returns a mask with only the requested lane bit set if it fits within 64 lanes.
    constexpr std::uint64_t singleBit(std::uint32_t lane)
    {
        if(lane >= 64u)
        {
            return 0u;
        }
        return std::uint64_t{1} << lane;
    }
} // namespace alpaka::test::warp
