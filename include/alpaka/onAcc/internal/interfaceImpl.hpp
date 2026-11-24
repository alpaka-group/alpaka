/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file
 * @brief This file contains specializations of interfaces for the accelerator scope.
 * The specializations must be separated from the definitions to avoid cyclic include dependencies.
 */

#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

namespace alpaka::onAcc
{
    namespace internalCompute
    {
        template<concepts::Acc T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::warp), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector<uint32_t, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::warp),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return Vec{warp::internal::getLaneIdx(acc)};
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::thread].idx();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<uint32_t, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                return Vec{warp::internal::getWarpIdx(acc)};
            }
        };

        template<typename T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<uint32_t, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                auto blockIdxInGrid = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
                auto numBlocksInGrid = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
                auto linearBlockIdx = linearize(numBlocksInGrid, blockIdxInGrid);

                std::integral auto linearNumWarpsInBlock
                    = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).product();
                return linearBlockIdx * linearNumWarpsInBlock + Vec{warp::internal::getWarpIdx(acc)};
            }
        };

        template<concepts::Acc T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::blocks)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::blocks)) const
            {
                return acc[layer::block].idx();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::thread), ALPAKA_TYPEOF(unit::threads)>
        {
            /** The identity of the thread.
             *
             * @return Zero for all components of the extent.
             */
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::thread),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                using ExtentType = ALPAKA_TYPEOF(acc[layer::thread].idx());

                using ValueType = typename ExtentType::value_type;
                constexpr uint32_t dim = ExtentType::dim();

                return fillCVec<ValueType, dim, 0u>();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::warp), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::CVector<uint32_t> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::warp),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                using IndexType = ALPAKA_TYPEOF(acc[layer::thread].count());
                return alpaka::Vec<IndexType, 1u>{warp::internal::getNumActiveLanes(acc)};
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::thread].count();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::block), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<alpaka::NotRequired, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                using IndexType = ALPAKA_TYPEOF(acc[layer::thread].count());
                return alpaka::Vec<IndexType, 1u>{warp::internal::getNumWarps(acc)};
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::blocks)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::blocks)) const
            {
                return acc[layer::block].count();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::warps)>
        {
            constexpr alpaka::concepts::Vector<alpaka::NotRequired, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                std::integral auto linearNumWarpsInBlock
                    = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).product();
                std::integral auto linearNumBlocks
                    = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks).product();
                return Vec{linearNumBlocks * linearNumWarpsInBlock};
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::grid), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return acc[layer::block].count() * acc[layer::thread].count();
            }
        };

        template<concepts::Acc T_Acc>
        struct GetExtentsOf::Op<T_Acc, ALPAKA_TYPEOF(origin::thread), ALPAKA_TYPEOF(unit::threads)>
        {
            /** The identity of the thread.
             *
             * @return One for all components of the extent.
             */
            constexpr alpaka::concepts::Vector auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::thread),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                using ExtentType = ALPAKA_TYPEOF(acc[layer::thread].count());
                using ValueType = typename ExtentType::value_type;
                constexpr uint32_t dim = ExtentType::dim();

                return fillCVec<ValueType, dim, 1u>();
            }
        };
    } // namespace internalCompute
} // namespace alpaka::onAcc
