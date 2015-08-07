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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_ACC_NO_CUDA

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The OpenMP barrier block synchronization.
            //#############################################################################
            class BlockSyncOmpBarrier
            {
            public:
                using BlockSyncBase = BlockSyncOmpBarrier;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncOmpBarrier() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncOmpBarrier(BlockSyncOmpBarrier const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSyncOmpBarrier(BlockSyncOmpBarrier &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncOmpBarrier const &) -> BlockSyncOmpBarrier & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSyncOmpBarrier &&) -> BlockSyncOmpBarrier & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSyncOmpBarrier() = default;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncOmpBarrier>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto syncBlockThreads(
                        block::sync::BlockSyncOmpBarrier const & blockSync)
                    -> void
                    {
                        boost::ignore_unused(blockSync);

                        // TODO: Use a barrier implementation not waiting for all threads:
                        // http://berenger.eu/blog/copenmp-custom-barrier-a-barrier-for-a-group-of-threads/
                        #pragma omp barrier
                    }
                };
            }
        }
    }
}
