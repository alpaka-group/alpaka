/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <type_traits>
#include <array>

#ifndef ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
#define ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB 30
#endif

namespace alpaka
{
    namespace block
    {
        namespace dyn
        {
            namespace detail
            {
                //#############################################################################
                //! "namespace" for static constexpr members that should be in BlockSharedMemDynMember
                //! but cannot be because having a static const member breaks GCC 10
                //! OpenMP target and OpenACC: type not mappable.
                template<std::size_t TStaticAllocKiB>
                struct BlockSharedMemDynMemberStatic
                {
                    //! Storage size in bytes
                    static constexpr std::size_t staticAllocBytes = TStaticAllocKiB<<10;
                };
            }

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#pragma warning(push)
#pragma warning(disable: 4324)  // warning C4324: structure was padded due to alignment specifier
#endif
            //#############################################################################
            //! Dynamic block shared memory provider using fixed-size
            //! member array to allocate memory on the stack or in shared
            //! memory.
            template<std::size_t TStaticAllocKiB = ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB>
            class alignas(core::vectorization::defaultAlignment) BlockSharedMemDynMember :
                public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynMember<TStaticAllocKiB>>
            {
            public:
                //-----------------------------------------------------------------------------
                BlockSharedMemDynMember(std::size_t sizeBytes)
                    : m_dynPitch((sizeBytes / core::vectorization::defaultAlignment
                            + (sizeBytes % core::vectorization::defaultAlignment > 0u)) * core::vectorization::defaultAlignment)
                {
#if (defined ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST) && (! defined NDEBUG)
                    ALPAKA_ASSERT(sizeBytes <= staticAllocBytes());
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

                /*! \return the pointer to the begin of data after the portion allocated as dynamical shared memory.
                    */
                uint8_t* staticMemBegin() const
                {
                    return m_mem.data() + m_dynPitch;
                }

                /*! \return the remaining capacity for static block shared memory.
                    */
                std::size_t staticMemCapacity() const
                {
                    return staticAllocBytes() - m_dynPitch;
                }

                //! Storage size in bytes
                static constexpr std::size_t staticAllocBytes() {return detail::BlockSharedMemDynMemberStatic<TStaticAllocKiB>::staticAllocBytes;}

            private:

                mutable std::array<uint8_t, detail::BlockSharedMemDynMemberStatic<TStaticAllocKiB>::staticAllocBytes> m_mem;
                std::size_t m_dynPitch;
            };
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#pragma warning(pop)
#endif

            namespace traits
            {
                //#############################################################################
                template<
                    typename T,
                    std::size_t TStaticAllocKiB>
                struct GetMem<
                    T,
                    BlockSharedMemDynMember<TStaticAllocKiB>>
                {
#if BOOST_COMP_GNUC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
                    //-----------------------------------------------------------------------------
                    static auto getMem(
                        block::dyn::BlockSharedMemDynMember<TStaticAllocKiB> const &mem)
                    -> T *
                    {
                        static_assert(
                            core::vectorization::defaultAlignment >= alignof(T),
                            "Unable to get block shared dynamic memory for types with alignment higher than defaultAlignment!");
                        return reinterpret_cast<T*>(mem.dynMemBegin());
                    }
#if BOOST_COMP_GNUC
#pragma GCC diagnostic pop
#endif
                };
            }
        }
    }
}
