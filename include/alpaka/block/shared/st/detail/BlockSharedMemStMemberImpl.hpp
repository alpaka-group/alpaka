/* Copyright 2020 Jeffrey Kelling, Rene Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Implementation of static block shared memory provider.
        //!
        //! externally allocated fixed-size memory, likely provided by BlockSharedMemDynMember.
        template<std::size_t TMinDataAlignBytes = core::vectorization::defaultAlignment>
        class BlockSharedMemStMemberImpl
        {
            struct MetaData
            {
                //! Unique id if the next data chunk.
                uint32_t id = std::numeric_limits<uint32_t>::max();
                //! Offset to the next meta data header, relative to m_mem.
                //! To access the meta data header the offset must by aligned first.
                uint32_t offset = 0;
            };

            static constexpr uint32_t metaDataSize = sizeof(MetaData);

        public:
            //-----------------------------------------------------------------------------
#ifndef NDEBUG
            BlockSharedMemStMemberImpl(uint8_t* mem, std::size_t capacity)
                : m_mem(mem)
                , m_capacity(static_cast<std::uint32_t>(capacity))
            {
                ALPAKA_ASSERT_OFFLOAD((m_mem == nullptr) == (m_capacity == 0u));
            }
#else
            BlockSharedMemStMemberImpl(uint8_t* mem, std::size_t) : m_mem(mem)
            {
            }
#endif
            //-----------------------------------------------------------------------------
            BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl const&) = delete;
            //-----------------------------------------------------------------------------
            BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl&&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(BlockSharedMemStMemberImpl const&) -> BlockSharedMemStMemberImpl& = delete;
            //-----------------------------------------------------------------------------
            auto operator=(BlockSharedMemStMemberImpl&&) -> BlockSharedMemStMemberImpl& = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~BlockSharedMemStMemberImpl() = default;

            template<typename T>
            void alloc(uint32_t id) const
            {
                // Add meta data chunk in front of the user data
                m_allocdBytes = varChunkEnd<MetaData>(m_allocdBytes);
                ALPAKA_ASSERT_OFFLOAD(m_allocdBytes <= m_capacity);
                MetaData* meta = getLatestVarPtr<MetaData>();

                // Allocate variable
                m_allocdBytes = varChunkEnd<T>(m_allocdBytes);
                ALPAKA_ASSERT_OFFLOAD(m_allocdBytes <= m_capacity);

                // Update meta data with id and offset for the allocated variable.
                meta->id = id;
                meta->offset = m_allocdBytes;
            }

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif

            //! Give the pointer to an exiting variable
            //!
            //! @tparam T type of the variable
            //! @param id unique id of the variable
            //! @return nullptr if variable with id not exists
            template<typename T>
            T* getVarPtr(uint32_t id) const
            {
                // Offset in bytes to the next unaligned meta data header behind the variable.
                uint32_t off = 0;

                // Iterate over allocated data only
                while(off < m_allocdBytes)
                {
                    // Adjust offset to be aligned
                    uint32_t const alignedMetaDataOffset
                        = varChunkEnd<MetaData>(off) - static_cast<uint32_t>(sizeof(MetaData));
                    ALPAKA_ASSERT_OFFLOAD(
                        (alignedMetaDataOffset + static_cast<uint32_t>(sizeof(MetaData))) <= m_allocdBytes);
                    MetaData* metaDataPtr = reinterpret_cast<MetaData*>(m_mem + alignedMetaDataOffset);
                    off = metaDataPtr->offset;

                    if(metaDataPtr->id == id)
                        return reinterpret_cast<T*>(&m_mem[off - sizeof(T)]);
                }

                // Variable not found.
                return nullptr;
            }

            //! Get last allocated variable.
            template<typename T>
            T* getLatestVarPtr() const
            {
                return reinterpret_cast<T*>(&m_mem[m_allocdBytes - sizeof(T)]);
            }

        private:
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

            //! Byte offset to the end of the memory chunk
            //!
            //! Calculate bytes required to store a type with a aligned starting address in m_mem.
            //! Start offset to the origin of the user data chunk can be calculated with `result - sizeof(T)`.
            //! The padding is always before the origin of the user data chunk and can be zero byte.
            //!
            //! \tparam T type should fit into the chunk
            //! \param byteOffset Current byte offset.
            //! \result Byte offset to the end of the data chunk, relative to m_mem..
            template<typename T>
            std::uint32_t varChunkEnd(uint32_t byteOffset) const
            {
                size_t const ptr = reinterpret_cast<size_t>(m_mem + byteOffset);
                constexpr size_t align = std::max(TMinDataAlignBytes, alignof(T));
                size_t const newPtrAdress = ((ptr + align - 1u) / align) * align + sizeof(T);
                return static_cast<uint32_t>(newPtrAdress - reinterpret_cast<size_t>(m_mem));
            }

            //! Offset in bytes relative to m_mem to next free data area.
            //! The last aligned before the free area is always a meta data header.
            mutable std::uint32_t m_allocdBytes = 0u;

            //! Memory layout
            //! |Header|Padding|Variable|Padding|Header|....uninitialized Data ....
            //! Size of padding can be zero if data after padding is already aligned.
            uint8_t* const m_mem;
#ifndef NDEBUG
            const std::uint32_t m_capacity;
#endif
        };
    } // namespace detail
} // namespace alpaka
