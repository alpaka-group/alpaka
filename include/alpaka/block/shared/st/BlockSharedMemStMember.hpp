/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/core/Assert.hpp>

#include <type_traits>
#include <cstdint>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                namespace detail
                {
                    //#############################################################################
                    //! Implementation of static block shared memory provider.
                    template<unsigned int TDataAlignBytes = 4>
                    class BlockSharedMemStMemberImpl
                    {
                        mutable unsigned int m_allocdBytes = 0;
                        mutable uint8_t* m_mem;
#ifndef NDEBUG
                        const unsigned int m_capacity;
#endif

                        template<typename T>
                        static constexpr size_t alignPitch()
                        {
                            return (sizeof(T)/TDataAlignBytes + (sizeof(T)%TDataAlignBytes>0))*TDataAlignBytes;
                        }

                    public:
                        //-----------------------------------------------------------------------------
#ifndef NDEBUG
                        BlockSharedMemStMemberImpl(uint8_t* mem, unsigned int capacity) : m_mem(mem), m_capacity(capacity) {}
#else
                        BlockSharedMemStMemberImpl(uint8_t* mem, unsigned int) : m_mem(mem) {}
#endif
                        //-----------------------------------------------------------------------------
                        BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl const &) = delete;
                        //-----------------------------------------------------------------------------
                        BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl &&) = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockSharedMemStMemberImpl const &) -> BlockSharedMemStMemberImpl & = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockSharedMemStMemberImpl &&) -> BlockSharedMemStMemberImpl & = delete;
                        //-----------------------------------------------------------------------------
                        /*virtual*/ ~BlockSharedMemStMemberImpl() = default;

                        template <typename T>
                        void alloc() const
                        {
                            uint8_t* buf = &m_mem[m_allocdBytes];
                            new (buf) T();
                            m_allocdBytes += alignPitch<T>();
                            ALPAKA_ASSERT(m_allocdBytes < m_capacity);
                        }

                        template <typename T>
                        T& getLatestVar() const
                        {
                           return *reinterpret_cast<T*>(&m_mem[m_allocdBytes-alignPitch<T>()]);
                        }

                        void free() const
                        {
                            m_allocdBytes = 0u;
                        }

                    };
                }
                //#############################################################################
                //! Static block shared memory provider using a pointer to
                //! externally allocated fixed-size memory, likely provided by
                //! BlockSharedMemDynMember.
                template<unsigned int TDataAlignBytes = 4>
                class BlockSharedMemStMember :
                    public detail::BlockSharedMemStMemberImpl<TDataAlignBytes>,
                    public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStMember<TDataAlignBytes>>
                {
                public:
                    using detail::BlockSharedMemStMemberImpl<TDataAlignBytes>::BlockSharedMemStMemberImpl;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        unsigned int TDataAlignBytes,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStMember<TDataAlignBytes>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto allocVar(
                            block::shared::st::BlockSharedMemStMember<TDataAlignBytes> const &smem)
                        -> T &
                        {
                            smem.template alloc<T>();
                            return smem.template getLatestVar<T>();
                        }
                    };
                    //#############################################################################
                    template<
                        unsigned int TDataAlignBytes>
                    struct FreeMem<
                        BlockSharedMemStMember<TDataAlignBytes>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto freeMem(
                            block::shared::st::BlockSharedMemStMember<TDataAlignBytes> const &mem)
                        -> void
                        {
                            mem.free();
                        }
                    };
                }
            }
        }
    }
}
