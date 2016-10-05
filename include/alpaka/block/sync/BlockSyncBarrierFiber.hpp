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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#include <alpaka/block/sync/Traits.hpp> // SyncBlockThread

#include <alpaka/core/Fibers.hpp>       // boost::fibers::barrier

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_ACC_NO_CUDA

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
            class BlockSyncBarrierFiber
            {
            public:
                using BlockSyncBase = BlockSyncBarrierFiber;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncBarrierFiber(
                    TSize const & blockThreadCount) :
                        m_barrier(static_cast<std::size_t>(blockThreadCount))
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncBarrierFiber(BlockSyncBarrierFiber const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncBarrierFiber(BlockSyncBarrierFiber &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncBarrierFiber const &) -> BlockSyncBarrierFiber & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncBarrierFiber &&) -> BlockSyncBarrierFiber & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSyncBarrierFiber() = default;

                boost::fibers::barrier mutable m_barrier;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TSize>
                struct SyncBlockThread<
                    BlockSyncBarrierFiber<TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierFiber<TSize> const & blockSync)
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
