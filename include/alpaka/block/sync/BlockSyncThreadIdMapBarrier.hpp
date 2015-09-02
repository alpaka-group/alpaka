/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/block/sync/Traits.hpp> // SyncBlockThreads

#include <alpaka/core/BarrierThread.hpp>// BarrierThread

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_ACC

#include <mutex>                        // std::mutex
#include <map>                          // std::map

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The thread id map barrier block synchronization.
            //#############################################################################
            template<
                typename TSize>
            class BlockSyncThreadIdMapBarrier
            {
            public:
                using BlockSyncBase = BlockSyncThreadIdMapBarrier;

                using Barrier = core::threads::BarrierThread<TSize>;
                using ThreadIdToBarrierIdxMap = std::map<std::thread::id, TSize>;
                using ThreadIdToBarrierIdxMapIterator = typename ThreadIdToBarrierIdxMap::iterator;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncThreadIdMapBarrier(
                    TSize const & numThreadsPerBlock,
                    ThreadIdToBarrierIdxMap & threadIdToBarrierIdxMap) :
                        m_threadsPerBlockCount(numThreadsPerBlock),
                        m_threadIdToBarrierIdxMap(threadIdToBarrierIdxMap)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncThreadIdMapBarrier(BlockSyncThreadIdMapBarrier const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncThreadIdMapBarrier(BlockSyncThreadIdMapBarrier &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncThreadIdMapBarrier const &) -> BlockSyncThreadIdMapBarrier & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncThreadIdMapBarrier &&) -> BlockSyncThreadIdMapBarrier & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSyncThreadIdMapBarrier() = default;

                //-----------------------------------------------------------------------------
                //! Syncs all threads in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads(
                    ThreadIdToBarrierIdxMapIterator const & itFind) const
                -> void
                {
                    assert(itFind != m_threadIdToBarrierIdxMap.end());

                    auto & barrierIdx(itFind->second);
                    TSize const modBarrierIdx(barrierIdx % 2);

                    auto & bar(m_barriers[modBarrierIdx]);

                    // (Re)initialize a barrier if this is the first thread to reach it.
                    // DCLP: Double checked locking pattern for better performance.
                    if(bar.getNumThreadsToWaitFor() == 0)
                    {
                        std::lock_guard<std::mutex> lock(m_mtxBarrier);
                        if(bar.getNumThreadsToWaitFor() == 0)
                        {
                            bar.reset(m_threadsPerBlockCount);
                        }
                    }

                    // Wait for the barrier.
                    bar.wait();
                    ++barrierIdx;
                }

                TSize const & m_threadsPerBlockCount;           //!< The number of threads per block the barrier has to wait for.

                ThreadIdToBarrierIdxMap & m_threadIdToBarrierIdxMap;
                //!< We have to keep the current and the last barrier because one of the threads can reach the next barrier before a other thread was wakeup from the last one and has checked if it can run.
                Barrier mutable m_barriers[2];           //!< The barriers for the synchronization of threads.
                std::mutex mutable m_mtxBarrier;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TSize>
                struct SyncBlockThreads<
                    BlockSyncThreadIdMapBarrier<TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto syncBlockThreads(
                        block::sync::BlockSyncThreadIdMapBarrier<TSize> const & blockSync)
                    -> void
                    {
                        auto const threadId(std::this_thread::get_id());
                        typename block::sync::BlockSyncThreadIdMapBarrier<TSize>::ThreadIdToBarrierIdxMapIterator const itFind(blockSync.m_threadIdToBarrierIdxMap.find(threadId));
                        blockSync.syncBlockThreads(itFind);
                    }
                };
            }
        }
    }
}
