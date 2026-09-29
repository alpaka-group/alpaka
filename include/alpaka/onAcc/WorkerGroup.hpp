/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/tag.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka::onAcc
{
    template<bool T_multiDimensional = true>
    struct MultiDimensional : std::bool_constant<T_multiDimensional>
    {
    };

    constexpr auto linearized = MultiDimensional<false>{};

    template<
        typename T_ThreadIdxOrOrigin,
        typename T_NumThreadsOrUnit,
        typename T_MultiDimensional = MultiDimensional<true>>
    struct WorkerGroup
    {
        /** WorkerGroup constructor
         *
         * @param threadIdxOrOrigin the index of the thread or onAcc::origin
         * @param numThreadsOrUnit the number of threads or the onAcc::unit
         * @param multiDimensional keep the dimensionality for both input parameters, if 'linearized' is used the
         * workgroup will be reduced to a one dimensional group.
         */
        constexpr WorkerGroup(
            T_ThreadIdxOrOrigin threadIdxOrOrigin,
            T_NumThreadsOrUnit numThreadsOrUnit,
            T_MultiDimensional = MultiDimensional<true>{})
            : m_threadIdxOrOrigin{threadIdxOrOrigin}
            , m_numThreadsOrUnit{numThreadsOrUnit}
        {
        }

        [[nodiscard]] constexpr auto getThreadCount(auto const& acc) const
        {
            return getThreadSpace(acc).getThreadCount();
        }

        [[nodiscard]] constexpr auto getThreadIdx(auto const& acc) const
        {
            return getThreadSpace(acc).getThreadIdx();
        }

    private:
        template<typename T_ThreadGroup, typename T_ThreadIdxOrOriginRange>
        friend struct DomainSpec;

        /** get the thread configuration
         *
         * Implementation specialization for vectors.
         */
        constexpr auto getThreadSpace([[maybe_unused]] auto const& acc) const
            requires(isVector_v<T_ThreadIdxOrOrigin> && isVector_v<T_NumThreadsOrUnit>)
        {
            if constexpr(T_MultiDimensional::value == false)
                return ThreadSpace{
                    Vec{linearize(m_numThreadsOrUnit, m_threadIdxOrOrigin)},
                    Vec{m_numThreadsOrUnit.product()}};
            else
                return ThreadSpace{m_threadIdxOrOrigin, m_numThreadsOrUnit};
        }

        /** get the thread configuration
         *
         * Implementation specialization for lazy evaluated acc properties based on an origin and unit.
         */
        constexpr auto getThreadSpace(auto const& acc) const
            requires(isOrigin_v<T_ThreadIdxOrOrigin> && isUnit_v<T_NumThreadsOrUnit>)
        {
            auto const idx
                = internalCompute::GetIdxWithin::Op<ALPAKA_TYPEOF(acc), T_ThreadIdxOrOrigin, T_NumThreadsOrUnit>{}(
                    acc,
                    m_threadIdxOrOrigin,
                    m_numThreadsOrUnit);
            auto const extent
                = internalCompute::GetExtentsOf::Op<ALPAKA_TYPEOF(acc), T_ThreadIdxOrOrigin, T_NumThreadsOrUnit>{}(
                    acc,
                    m_threadIdxOrOrigin,
                    m_numThreadsOrUnit);

            if constexpr(T_MultiDimensional::value == false)
                return ThreadSpace{Vec{linearize(extent, idx)}, Vec{extent.product()}};
            else
                return ThreadSpace{idx, extent};
        }

    private:
        T_ThreadIdxOrOrigin m_threadIdxOrOrigin;
        T_NumThreadsOrUnit m_numThreadsOrUnit;
    };

    namespace worker
    {
        constexpr auto threadsInGrid = WorkerGroup{origin::grid, unit::threads};
        constexpr auto blocksInGrid = WorkerGroup{origin::grid, unit::blocks};
        constexpr auto threadsInBlock = WorkerGroup{origin::block, unit::threads};

        /** Representation of all threads in a warp as a linearized worker group */
        constexpr auto linearThreadsInWarp = WorkerGroup{origin::warp, unit::threads};
        constexpr auto linearThreadsInBlock = WorkerGroup{origin::block, unit::threads, linearized};
        constexpr auto linearThreadsInGrid = WorkerGroup{origin::grid, unit::threads, linearized};

        /** Representation of all warps in a thread block as a linearized worker group
         *
         * @attention If the number of threads in a block is not a multiple of the warp size you can have partial
         * warps.
         */
        constexpr auto linearWarpsInBlock = WorkerGroup{origin::block, unit::warps};
        /** Representation of all warps in the grid as a linearized worker group
         *
         * @attention Since a thread block is not required to have as many threads as a warp has, you can not assume
         * that number of warps * warp size is the total number of threads.
         */
        constexpr auto linearWarpsInGrid = WorkerGroup{origin::grid, unit::warps};

        constexpr auto linearBlocksInGrid = WorkerGroup{origin::grid, unit::blocks, linearized};

        /** Represent the identity of the executor thread.
         *
         * All threads are in the same worker group.
         * If used with onAcc::makeIdxMap(), any thread is getting all indices of the range.
         */
        constexpr auto allThreads = WorkerGroup{origin::thread, unit::threads};
    } // namespace worker

} // namespace alpaka::onAcc
