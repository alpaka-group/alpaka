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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#include <alpaka/block/sync/Traits.hpp> // SyncBlockThread

#include <alpaka/core/BarrierThread.hpp>// BarrierThread

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_ACC

#include <thread>                       // std::thread
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
            class BlockSyncBarrierThread
            {
            public:
                using BlockSyncBase = BlockSyncBarrierThread;

                using Barrier = core::threads::BarrierThread<TSize>;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncBarrierThread(
                    TSize const & blockThreadCount) :
                        m_barrier(blockThreadCount)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncBarrierThread(BlockSyncBarrierThread const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncBarrierThread(BlockSyncBarrierThread &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncBarrierThread const &) -> BlockSyncBarrierThread & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncBarrierThread &&) -> BlockSyncBarrierThread & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSyncBarrierThread() = default;

                Barrier mutable m_barrier;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TSize>
                struct SyncBlockThread<
                    BlockSyncBarrierThread<TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierThread<TSize> const & blockSync)
                    -> void
                    {
                        blockSync.m_barrier.wait();
                    }
                };
            }
        }
    }
}

#endif
