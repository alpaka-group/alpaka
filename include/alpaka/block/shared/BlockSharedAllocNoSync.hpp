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

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            //#############################################################################
            //! The block shared memory allocator without synchronization.
            //#############################################################################
            class BlockSharedAllocNoSync
            {
            public:
                using BlockSharedAllocBase = BlockSharedAllocNoSync;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSharedAllocNoSync() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSharedAllocNoSync(BlockSharedAllocNoSync const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BlockSharedAllocNoSync(BlockSharedAllocNoSync &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedAllocNoSync const &) -> BlockSharedAllocNoSync & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedAllocNoSync &&) -> BlockSharedAllocNoSync & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSharedAllocNoSync() = default;

            public:
                // TODO: We should add the size of the (current) allocation.
                // This would allow to assert that all parallel function calls request to allocate the same size.
                std::vector<
                    std::unique_ptr<
                        uint8_t,
                        boost::alignment::aligned_delete>> mutable
                    m_sharedAllocs;    //!< Block shared memory.
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
                    BlockSharedAllocNoSync>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto allocVar(
                        block::shared::BlockSharedAllocNoSync const & blockSharedAlloc)
                    -> T &
                    {
                        blockSharedAlloc.m_sharedAllocs.emplace_back(
                            reinterpret_cast<uint8_t *>(
                                boost::alignment::aligned_alloc(16u, sizeof(T))));
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
                    BlockSharedAllocNoSync>
                {
                    static_assert(
                        TnumElements > 0,
                        "The number of elements to allocate in block shared memory must not be zero!");

                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto allocArr(
                        block::shared::BlockSharedAllocNoSync const & blockSharedAlloc)
                    -> T *
                    {
                        blockSharedAlloc.m_sharedAllocs.emplace_back(
                            reinterpret_cast<uint8_t *>(
                                boost::alignment::aligned_alloc(16u, sizeof(T) * TnumElements)));
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
                    BlockSharedAllocNoSync>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA static auto freeMem(
                        block::shared::BlockSharedAllocNoSync const & blockSharedAlloc)
                    -> void
                    {
                        blockSharedAlloc.m_sharedAllocs.clear();
                    }
                };
            }
        }
    }
}
