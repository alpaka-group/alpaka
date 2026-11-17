/* Copyright 2022 Jeffrey Kelling, Rene Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#ifndef ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
#    define ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB 64u
#endif

namespace alpaka::onAcc::cpu::detail
{
    //! Implementation of static block shared memory provider.
    //!
    //! externally allocated fixed-size memory, likely provided by BlockSharedMemDynMember.
    template<std::size_t T_minDataAlignBytes>
    class SharedStorage
    {
        struct alignas(T_minDataAlignBytes) MetaData
        {
            //! Unique id if the next data chunk.
            size_t id = 0u;
            //! Offset to the next meta data header, relative to m_mem.
            //! To access the meta data header the offset must by aligned first.
            std::uint32_t offset = 0u;
        };

        static constexpr std::uint32_t metaDataSize = sizeof(MetaData);

    public:
        SharedStorage() = default;

        template<typename T>
        void alloc(size_t id) const
        {
            // Add meta data chunk in front of the user data
            m_allocdBytes = varChunkEnd<MetaData>(m_allocdBytes, sizeof(MetaData));
            ALPAKA_ASSERT_ACC(m_allocdBytes <= totalSharedBytes);
            auto* meta = getLatestVarPtr<MetaData>();

            // Allocate variable
            m_allocdBytes = varChunkEnd<T>(m_allocdBytes, sizeof(T));
            ALPAKA_ASSERT_ACC(m_allocdBytes <= totalSharedBytes);

            // Update meta data with id and offset for the allocated variable.
            meta->id = id;
            meta->offset = m_allocdBytes;
        }

        template<typename T>
        void allocDynamic(size_t id, uint32_t numBytes) const
        {
            // Add meta data chunk in front of the user data
            m_allocdBytes = varChunkEnd<MetaData>(m_allocdBytes, sizeof(MetaData));
            ALPAKA_ASSERT_ACC(m_allocdBytes <= totalSharedBytes);
            auto* meta = getLatestVarPtr<MetaData>();

            // Allocate variable
            m_allocdBytes = varChunkEnd<T>(m_allocdBytes, numBytes);
            ALPAKA_ASSERT_ACC(m_allocdBytes <= totalSharedBytes);

            // Update meta data with id and offset for the allocated variable.
            meta->id = id;
            meta->offset = m_allocdBytes;
        }

        //! Give the pointer to an exiting variable
        //!
        //! @tparam T type of the variable
        //! @param id unique id of the variable
        //! @return nullptr if variable with id not exists
        template<typename T>
        auto getVarPtr(size_t id) const -> T*
        {
            // Offset in bytes to the next unaligned meta data header behind the variable.
            std::uint32_t off = 0;

            // Iterate over allocated data only
            while(off < m_allocdBytes)
            {
                // Adjust offset to be aligned
                std::uint32_t const alignedMetaDataOffset
                    = varChunkEnd<MetaData>(off, sizeof(MetaData)) - static_cast<std::uint32_t>(sizeof(MetaData));
                ALPAKA_ASSERT_ACC(
                    (alignedMetaDataOffset + static_cast<std::uint32_t>(sizeof(MetaData))) <= m_allocdBytes);
                auto* metaDataPtr = reinterpret_cast<MetaData*>(data() + alignedMetaDataOffset);
                off = metaDataPtr->offset;

                if(metaDataPtr->id == id)
                    return reinterpret_cast<T*>(&data()[off - sizeof(T)]);
            }

            // Variable not found.
            return nullptr;
        }

        //! Get last allocated variable.
        template<typename T>
        auto getLatestVarPtr() const -> T*
        {
            return reinterpret_cast<T*>(&data()[m_allocdBytes - sizeof(T)]);
        }

    private:
        uint8_t* data() const
        {
            return m_data.data();
        }

        //! Byte offset to the end of the memory chunk
        //!
        //! Calculate bytes required to store a type with a aligned starting address in m_mem.
        //! Start offset to the origin of the user data chunk can be calculated with `result - sizeof(T)`.
        //! The padding is always before the origin of the user data chunk and can be zero byte.
        //!
        //! \tparam T type should fit into the chunk
        //! \param byteOffset Current byte offset.
        //! \param numBytes Number of bytes to allocate, should be at least sizeof(T).
        //! \result Byte offset to the end of the data chunk, relative to m_mem..
        template<typename T>
        auto varChunkEnd(uint32_t byteOffset, uint32_t numBytes) const -> std::uint32_t
        {
            auto const ptr = reinterpret_cast<std::size_t>(data() + byteOffset);
            constexpr size_t align = std::max(T_minDataAlignBytes, alignof(T));
            std::size_t const newPtrAdress = ((ptr + align - 1u) / align) * align + numBytes;
            return static_cast<uint32_t>(newPtrAdress - reinterpret_cast<std::size_t>(data()));
        }

        static constexpr std::uint32_t totalSharedBytes
            = static_cast<std::uint32_t>(ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB << 10u);
        //! Memory layout
        //! |Header|Padding|Variable|Padding|Header|....uninitialized Data ....
        //! Size of padding can be zero if data after padding is already aligned.
        mutable std::array<uint8_t, totalSharedBytes> m_data;

        //! Offset in bytes relative to m_mem to next free data area.
        //! The last aligned before the free area is always a meta data header.
        mutable std::uint32_t m_allocdBytes = 0u;
    };
} // namespace alpaka::onAcc::cpu::detail
