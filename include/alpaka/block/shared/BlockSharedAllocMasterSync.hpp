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

#include <alpaka/block/shared/Traits.hpp>   // AllocVar, AllocArr

#include <alpaka/core/Common.hpp>           // ALPAKA_FN_ACC_NO_CUDA

#include <boost/align.hpp>                  // boost::aligned_alloc

#include <vector>                           // std::vector
#include <memory>                           // std::unique_ptr
#include <functional>                       // std::function

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            //#############################################################################
            //! The block shared memory allocator allocating memory with synchronization on the master thread.
            //#############################################################################
            class BlockSharedAllocMasterSync
            {
            public:
                using BlockSharedAllocBase = BlockSharedAllocMasterSync;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSharedAllocMasterSync(
                    std::function<void()> fnSync,
                    std::function<bool()> fnIsMasterThread) :
                        m_syncFn(fnSync),
                        m_isMasterThreadFn(fnIsMasterThread)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSharedAllocMasterSync(BlockSharedAllocMasterSync const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSharedAllocMasterSync(BlockSharedAllocMasterSync &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedAllocMasterSync const &) -> BlockSharedAllocMasterSync & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedAllocMasterSync &&) -> BlockSharedAllocMasterSync & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSharedAllocMasterSync() = default;

            public:
                // TODO: We should add the size of the (current) allocation.
                // This would allow to assert that all parallel function calls request to allocate the same size.
                std::vector<
                    std::unique_ptr<
                        uint8_t,
                        boost::alignment::aligned_delete>> mutable
                    m_sharedAllocs;

                std::function<void()> m_syncFn;
                std::function<bool()> m_isMasterThreadFn;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename T>
                struct AllocVar<
                    T,
                    BlockSharedAllocMasterSync>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto allocVar(
                        block::shared::BlockSharedAllocMasterSync const & blockSharedAlloc)
                    -> T &
                    {
                        // Assure that all threads have executed the return of the last allocBlockSharedArr function (if there was one before).
                        blockSharedAlloc.m_syncFn();

                        // Arbitrary decision: The fiber that was created first has to allocate the memory.
                        if(blockSharedAlloc.m_isMasterThreadFn())
                        {
                            blockSharedAlloc.m_sharedAllocs.emplace_back(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, sizeof(T))));
                        }
                        blockSharedAlloc.m_syncFn();

                        return
                            std::ref(
                                *reinterpret_cast<T*>(
                                    blockSharedAlloc.m_sharedAllocs.back().get()));
                    }
                };
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename T,
                    std::size_t TnumElements>
                struct AllocArr<
                    T,
                    TnumElements,
                    BlockSharedAllocMasterSync>
                {
                    static_assert(
                        TnumElements > 0,
                        "The number of elements to allocate in block shared memory must not be zero!");

                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto allocArr(
                        block::shared::BlockSharedAllocMasterSync const & blockSharedAlloc)
                    -> T *
                    {
                        // Assure that all threads have executed the return of the last allocBlockSharedArr function (if there was one before).
                        blockSharedAlloc.m_syncFn();

                        // Arbitrary decision: The fiber that was created first has to allocate the memory.
                        if(blockSharedAlloc.m_isMasterThreadFn())
                        {
                            blockSharedAlloc.m_sharedAllocs.emplace_back(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, sizeof(T) * TnumElements)));
                        }
                        blockSharedAlloc.m_syncFn();

                        return
                            reinterpret_cast<T*>(
                                blockSharedAlloc.m_sharedAllocs.back().get());
                    }
                };
                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct FreeMem<
                    BlockSharedAllocMasterSync>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto freeMem(
                        block::shared::BlockSharedAllocMasterSync const & blockSharedAlloc)
                    -> void
                    {
                        blockSharedAlloc.m_sharedAllocs.clear();
                    }
                };
            }
        }
    }
}
