/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file This file contains specializations of interfaces for the accelerator scope.
 * The specializations must be separated from the definitions to avoid cyclic include dependencies.
 */

#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <utility>

namespace alpaka::onAcc
{
    namespace internalCompute
    {
        /** Index type of the threads in a block, used for all warp indices and extents. */
        template<typename T_Acc>
        using BlockIdxType
            = alpaka::GetValueType_t<ALPAKA_TYPEOF(std::declval<T_Acc const&>()[layer::thread].count())>;

        template<concepts::Acc T_Acc>
        struct GetIdxWithin::Op<T_Acc, ALPAKA_TYPEOF(origin::warp), ALPAKA_TYPEOF(unit::threads)>
        {
            constexpr alpaka::concepts::Vector<BlockIdxType<T_Acc>, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::warp),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                return Vec{static_cast<BlockIdxType<T_Acc>>(warp::internal::getLaneIdx(acc))};
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
            constexpr alpaka::concepts::Vector<BlockIdxType<T_Acc>, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::block),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                return Vec{static_cast<BlockIdxType<T_Acc>>(warp::internal::getWarpIdx(acc))};
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
            constexpr alpaka::concepts::Vector<alpaka::NotRequired, 1u> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::grid),
                ALPAKA_TYPEOF(unit::warps)) const
            {
                auto blockIdxInGrid = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
                auto numBlocksInGrid = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
                auto linearBlockIdx = linearize(numBlocksInGrid, blockIdxInGrid);

                std::integral auto linearNumWarpsInBlock
                    = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).product();
                using IndexType = alpaka::GetValueType_t<ALPAKA_TYPEOF(linearBlockIdx)>;
                return linearBlockIdx * static_cast<IndexType>(linearNumWarpsInBlock)
                       + Vec{static_cast<IndexType>(warp::internal::getWarpIdx(acc))};
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
            constexpr alpaka::concepts::CVector<BlockIdxType<T_Acc>> auto operator()(
                T_Acc const& acc,
                ALPAKA_TYPEOF(origin::warp),
                ALPAKA_TYPEOF(unit::threads)) const
            {
                alpaka::unused(acc);
                return alpaka::CVec<BlockIdxType<T_Acc>, T_Acc::getWarpSize()>{};
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
                std::integral auto linearThreadsInBlock
                    = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads).product();
                using IndexType = alpaka::GetValueType_t<ALPAKA_TYPEOF(linearThreadsInBlock)>;
                return Vec{divCeil(linearThreadsInBlock, static_cast<IndexType>(T_Acc::getWarpSize()))};
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
