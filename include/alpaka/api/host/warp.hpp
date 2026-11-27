/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Provides warp trait fallbacks for scalar host execution.
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp::internal
{
    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Activemask::Op<T_Acc, api::Host>
    {
        constexpr auto operator()(T_Acc const& acc, api::Host) const
        {
            alpaka::unused(acc);
            return uint32_t{1u};
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetLaneIdx::Op<T_Acc, api::Host>
    {
        constexpr auto operator()(T_Acc const& acc, api::Host) const
        {
            alpaka::unused(acc);
            return uint32_t{0u};
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetNumActiveLanes::Op<T_Acc, api::Host>
    {
        constexpr auto operator()(T_Acc const& acc, api::Host) const
        {
            alpaka::unused(acc);
            return uint32_t{1u};
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetWarpIdx::Op<T_Acc, api::Host>
    {
        constexpr auto operator()(T_Acc const& acc, api::Host) const
        {
            alpaka::unused(acc);
            return uint32_t{0u};
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetNumWarps::Op<T_Acc, api::Host>
    {
        constexpr auto operator()(T_Acc const& acc, api::Host) const
        {
            alpaka::unused(acc);
            return uint32_t{1u};
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct All::Op<T_Acc, api::Host>
    {
        constexpr bool operator()(T_Acc const& acc, api::Host, int32_t predicate) const
        {
            alpaka::unused(acc);
            return predicate != 0;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Any::Op<T_Acc, api::Host>
    {
        constexpr bool operator()(T_Acc const& acc, api::Host, int32_t predicate) const
        {
            alpaka::unused(acc);
            return predicate != 0;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Ballot::Op<T_Acc, api::Host>
    {
        constexpr auto operator()(T_Acc const& acc, api::Host, int32_t predicate) const
        {
            alpaka::unused(acc);
            return predicate != 0 ? 1u : 0u;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct Shfl::Op<T_Acc, api::Host, T>
    {
        constexpr T operator()(T_Acc const& acc, api::Host, T const& value, uint32_t srcLane, uint32_t width) const
        {
            alpaka::unused(acc, srcLane, width);
            return value;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflDown::Op<T_Acc, api::Host, T>
    {
        constexpr T operator()(T_Acc const& acc, api::Host, T const& value, uint32_t delta, uint32_t width) const
        {
            alpaka::unused(acc, delta, width);
            return value;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflUp::Op<T_Acc, api::Host, T>
    {
        constexpr T operator()(T_Acc const& acc, api::Host, T const& value, uint32_t delta, uint32_t width) const
        {
            alpaka::unused(acc, delta, width);
            return value;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflXor::Op<T_Acc, api::Host, T>
    {
        constexpr T operator()(T_Acc const& acc, api::Host, T const& value, uint32_t laneMask, uint32_t width) const
        {
            alpaka::unused(acc, laneMask, width);
            return value;
        }
    };
} // namespace alpaka::onAcc::warp::internal
