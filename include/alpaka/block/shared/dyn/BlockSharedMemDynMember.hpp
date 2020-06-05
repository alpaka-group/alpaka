/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/core/Assert.hpp>

#include <type_traits>
#include <array>

#ifndef ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KB
#define ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KB 30
#endif

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! Dynamic block shared memory provider using fixed-size
                //! member array to allocate memory on the stack or in shared
                //! memory.
                template<unsigned int TStaticAllocKB = ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KB>
                class BlockSharedMemDynMember :
                    public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynMember<TStaticAllocKB>>
                {
                    static constexpr unsigned int staticAllocBytes = TStaticAllocKB<<10;

                    mutable std::array<uint8_t, staticAllocBytes> m_mem;
                    unsigned int m_dynSize;

                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynMember(unsigned int sizeBytes) : m_dynSize(sizeBytes)
                    {
#if (defined ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST) && (! defined NDEBUG)
                        ALPAKA_ASSERT(sizeBytes <= staticAllocBytes);
#endif
                    }
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynMember(BlockSharedMemDynMember const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynMember(BlockSharedMemDynMember &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynMember const &) -> BlockSharedMemDynMember & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynMember &&) -> BlockSharedMemDynMember & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynMember() = default;

                    uint8_t* dynMemBegin() const {return m_mem.data();}

                    /*! \return the pointer to the begin of data after protion allocated as dynamical shared memory.
                     *!
                     *! \tparam TDataAlignBytes Round-up offset to multiple of
                     *! this number of bytes to maintain alignment. This
                     *! alignment assumes, that the compiler places the instance
                     *! at architecture-appropriate boundaries.
                     */
                    template<unsigned int TDataAlignBytes = 4>
                    uint8_t* staticMemBegin() const
                    {
                        return m_mem.data() +
                            (m_dynSize/TDataAlignBytes + (m_dynSize%TDataAlignBytes>0)*TDataAlignBytes);
                    }

                    /*! \return the remaining capacity for static block shared memory.
                     *!
                     *! \tparam TDataAlignBytes Multiple of bytes the static offset was rounded-up to
                     */
                    template<unsigned int TDataAlignBytes = 4>
                    unsigned int staticMemCapacity() const
                    {
                        return staticAllocBytes -
                            (m_dynSize/TDataAlignBytes + (m_dynSize%TDataAlignBytes>0)*TDataAlignBytes);
                    }
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        unsigned int TStaticAllocKB>
                    struct GetMem<
                        T,
                        BlockSharedMemDynMember<TStaticAllocKB>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            block::shared::dyn::BlockSharedMemDynMember<TStaticAllocKB> const &mem)
                        -> T *
                        {
                            return reinterpret_cast<T*>(mem.dynMemBegin());
                        }
                    };
                }
            }
        }
    }
}
