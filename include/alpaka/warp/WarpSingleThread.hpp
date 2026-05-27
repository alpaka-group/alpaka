/* Copyright 2026 Sergei Bastrakov, David M. Rogers, Bernhard Manfred Gruber, Aurora Perego, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/warp/Traits.hpp"

#include <cstdint>

namespace alpaka::warp
{
    //! The single-threaded warp to emulate it on CPUs.
    struct WarpSingleThread : public interface::Implements<ConceptWarp, WarpSingleThread>
    {
        using mask_type = std::uint32_t;
    };

    namespace trait
    {

        template<>
        struct GetSize<WarpSingleThread>
        {
            static auto getSize(warp::WarpSingleThread const& /*warp*/) -> std::int32_t
            {
                return 1;
            }
        };

        template<>
        struct GetSizeCompileTime<WarpSingleThread>
        {
            static constexpr auto getSizeCompileTime() -> std::int32_t
            {
                return 1;
            }
        };

        template<>
        struct GetSizeUpperLimit<WarpSingleThread>
        {
            static constexpr auto getSizeUpperLimit() -> std::int32_t
            {
                return 1;
            }
        };

        template<>
        struct Activemask<WarpSingleThread>
        {
            static auto activemask(warp::WarpSingleThread const& /*warp*/) -> WarpSingleThread::mask_type
            {
                return 1u;
            }
        };

        template<>
        struct All<WarpSingleThread>
        {
            static auto all(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
            {
                return predicate;
            }
        };

        template<>
        struct Any<WarpSingleThread>
        {
            static auto any(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
            {
                return predicate;
            }
        };

        template<>
        struct Ballot<WarpSingleThread>
        {
            static auto ballot(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
                -> WarpSingleThread::mask_type
            {
                return predicate ? 1u : 0u;
            }

            static auto ballot(warp::WarpSingleThread const& /*warp*/, WarpSingleThread::mask_type mask, std::int32_t predicate)
                -> WarpSingleThread::mask_type
            {
                return (predicate ? 1u : 0u) & mask;
            }
        };

        template<>
        struct Shfl<WarpSingleThread>
        {
            template<typename T>
            static auto shfl(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::int32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }

            template<typename T>
            static auto shfl(
                warp::WarpSingleThread const& /*warp*/,
                WarpSingleThread::mask_type mask,
                T val,
                std::int32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct ShflUp<WarpSingleThread>
        {
            template<typename T>
            static auto shfl_up(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::uint32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }

            template<typename T>
            static auto shfl_up(
                warp::WarpSingleThread const& /*warp*/,
                WarpSingleThread::mask_type mask,
                T val,
                std::uint32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct ShflDown<WarpSingleThread>
        {
            template<typename T>
            static auto shfl_down(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::uint32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }

            template<typename T>
            static auto shfl_down(
                warp::WarpSingleThread const& /*warp*/,
                WarpSingleThread::mask_type mask,
                T val,
                std::uint32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct ShflXor<WarpSingleThread>
        {
            template<typename T>
            static auto shfl_xor(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::int32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct MatchAny<WarpSingleThread>
        {
            static auto match_any(warp::WarpSingleThread const& /*warp*/, WarpSingleThread::mask_type mask, T val)
                -> WarpSingleThread::mask_type
            {
                return mask;
            }
        };

        template<>
        struct Brev<WarpSingleThread>
        {
            static auto brev(
                [[maybe_unused]] warp::WarpSingleThread const& warp,
                WarpSingleThread::mask_type mask
                ) -> WarpSingleThread::mask_type
            {
                return mask;
            }
        };

        template<>
        struct Clz<WarpSingleThread>
        {
            static auto clz(
                [[maybe_unused]] warp::WarpSingleThread const& warp,
                WarpSingleThread::mask_type mask
                ) -> std::uint32_t
            {
                return mask == 0 ? 1 : 0;
            }
        }; 
        
        template<>
        struct SyncWarpThreads<WarpSingleThread>
        {
            static auto syncWarpThreads(
                [[maybe_unused]] warp::WarpSingleThread const& warp,
                WarpSingleThread::mask_type mask
                ) -> void
            {
            }
        };           



    } // namespace trait
} // namespace alpaka::warp
