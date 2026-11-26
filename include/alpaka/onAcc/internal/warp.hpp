/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Defines the Alpaka3 warp traits for kernel calls for votes and shuffles.
 * Central helpers (all, any, ballot, shfl…) forward to trait::Op specializations.
 *
 * Dispatch now depends on api::X and deviceKind::Y tags, keeping the layer backend free.
 * Provides a single entry point that works for host and GPU backends alike.
 *
 * Legacy Alpaka preferred the dispatch through ConceptWarp implementations per accelerator and used
 * interface::ImplementationBase indirection to select the correct implementation.
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp::internal
{
    template<alpaka::onAcc::concepts::Acc T_Acc>
    constexpr uint32_t getSize()
    {
        return T_Acc::getWarpSize();
    }

    /** Retrieve a bit-mask describing which warp lanes are active. */
    struct Activemask
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp Activemask implementation for the accelerator.");
                return 0u;
            }
        };
    };

    struct GetLaneIdx
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp GetLaneIdx implementation for the accelerator.");
                return 0u;
            }
        };
    };

    /** Return the lane index of the current thread within its warp. */
    constexpr uint32_t getLaneIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return GetLaneIdx::Op<Acc, Api>{}(acc, Api{});
    }

    struct GetNumActiveLanes
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp GetNumActiveLanes implementation for the accelerator.");
                return 0u;
            }
        };
    };

    /** Return the number of lanes within the warp of the current thread, at most warp::getSize(). */
    constexpr uint32_t getNumActiveLanes(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return GetNumActiveLanes::Op<Acc, Api>{}(acc, Api{});
    }

    struct GetWarpIdx
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp GetWarpIdx implementation for the accelerator.");
                return 0u;
            }
        };
    };

    /** Return the warp index within the block. */
    constexpr uint32_t getWarpIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return GetWarpIdx::Op<Acc, Api>{}(acc, Api{});
    }

    struct GetNumWarps
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp GetNumWarps implementation for the accelerator.");
                return 0u;
            }
        };
    };

    /** Return the number of warps within the block. */
    constexpr uint32_t getNumWarps(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return GetNumWarps::Op<Acc, Api>{}(acc, Api{});
    }

    struct All
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr bool operator()(T_Acc const&, T_Api, int32_t predicate) const
            {
                alpaka::unused(predicate);
                static_assert(sizeof(T_Acc) && false, "Missing warp All implementation for the accelerator.");
                return false;
            }
        };
    };

    struct Any
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr bool operator()(T_Acc const&, T_Api, int32_t predicate) const
            {
                alpaka::unused(predicate);
                static_assert(sizeof(T_Acc) && false, "Missing warp Any implementation for the accelerator.");
                return false;
            }
        };
    };

    struct Ballot
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api, int32_t predicate) const
            {
                alpaka::unused(predicate);
                static_assert(sizeof(T_Acc) && false, "Missing warp Ballot implementation for the accelerator.");
                return 0;
            }
        };
    };

    struct Shfl
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t srcLane, uint32_t width) const
            {
                alpaka::unused(value, srcLane, width);
                static_assert(sizeof(T_Acc) && false, "Missing warp Shfl implementation for the accelerator.");
                return T{};
            }
        };
    };

    struct ShflDown
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t delta, uint32_t width) const
            {
                alpaka::unused(value, delta, width);
                static_assert(sizeof(T_Acc) && false, "Missing warp ShflDown implementation for the accelerator.");
                return T{};
            }
        };
    };

    struct ShflUp
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t delta, uint32_t width) const
            {
                alpaka::unused(value, delta, width);
                static_assert(sizeof(T_Acc) && false, "Missing warp ShflUp implementation for the accelerator.");
                return T{};
            }
        };
    };

    struct ShflXor
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t laneMask, uint32_t width) const
            {
                alpaka::unused(value, laneMask, width);
                static_assert(sizeof(T_Acc) && false, "Missing warp ShflXor implementation for the accelerator.");
                return T{};
            }
        };
    };
} // namespace alpaka::onAcc::warp::internal
