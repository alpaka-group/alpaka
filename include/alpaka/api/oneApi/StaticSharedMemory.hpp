/* Copyright 2025 Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/tag.hpp"

#if ALPAKA_LANG_ONEAPI

#    include <sycl/sycl.hpp>

#    include <functional>

namespace alpaka::onAcc
{
    namespace oneApi
    {
        namespace detail
        {
            /** Pointer lookup table
             *
             * Provides a dynamic lookup table to map a unique id to a pointer.
             */
            class PtrLookupTable
            {
                struct MetaData
                {
                    //! pointer to allocated data
                    std::byte* ptr = nullptr;
                    //! Unique id of the next data chunk.
                    size_t id = std::numeric_limits<size_t>::max();
                };

                static constexpr uint32_t metaDataSize = sizeof(MetaData);

            public:
#    ifndef NDEBUG
                PtrLookupTable(std::byte* mem, uint32_t capacity)
                    : m_mem(reinterpret_cast<MetaData*>(mem))
                    , m_capacity(capacity / metaDataSize)
                {
                    ALPAKA_ASSERT_ACC((m_mem == nullptr) == (m_capacity == 0u));
                }
#    else
                PtrLookupTable(std::byte* mem, uint32_t) : m_mem(reinterpret_cast<MetaData*>(mem))
                {
                }
#    endif

                /** number of bytes required for bookkeeping of maxNumberOfAllocations unique allocations
                 *
                 * @param maxNumUniqueAllocations number of unique allocation a user is allowed to perform
                 * @return bytes required to store lookup meta data
                 */
                static consteval uint32_t sizeLookupBufferInBytes(uint32_t maxNumUniqueAllocations)
                {
                    return metaDataSize * maxNumUniqueAllocations;
                }

                /* With oneApi 2025.2 the behaviour of shared memory allocation has changed. It behaves like cuda
                 * shared memory. Therefore, we need a unique data type to avoid pointer aliasing. Using the helper
                 * class for data alignment is backward compatible to previous versions. The reason for using std::byte
                 * is that this guaranteed support for data types which are not trivially constructible.
                 */
                template<typename T, size_t T_id>
                struct alignas(T) SharedMemData
                {
                    std::byte data[sizeof(T)];
                };

                template<typename T, size_t T_id>
                T* alloc() const
                {
                    auto group = sycl::ext::oneapi::this_work_item::get_work_group<1>();
                    SharedMemData<T, T_id>* data
                        = sycl::ext::oneapi::group_local_memory_for_overwrite<SharedMemData<T, T_id>>(group);

                    MetaData& metaDataEntry = m_mem[m_numEntries];
                    ++m_numEntries;
                    ALPAKA_ASSERT_ACC(m_numEntries <= m_capacity);

                    // Update meta data with id and pointer to the current allocation
                    if(group.get_local_linear_id() == 0u)
                    {
                        // only one thread must update the pointer in shared memory
                        metaDataEntry.ptr = reinterpret_cast<std::byte*>(data);
                    }
                    metaDataEntry.id = T_id;

                    return reinterpret_cast<T*>(data);
                }

                //! Give the pointer to an existing variable
                //!
                //! @tparam T type of the variable
                //! @param id unique id of the variable
                //! @return nullptr if variable with id does not exist
                template<typename T>
                auto getVarPtr(size_t id) const -> T*
                {
                    // Iterate over metadata
                    for(uint32_t off = 0u; off < m_numEntries; ++off)
                    {
                        MetaData& metaDataEntry = m_mem[off];

                        if(metaDataEntry.id == id)
                            return reinterpret_cast<T*>(metaDataEntry.ptr);
                    }

                    // Variable not found.
                    return nullptr;
                }

            private:
                //! Number unique meta data entries stored
                mutable uint32_t m_numEntries = 0u;

                //! Memory layout
                //! |Header|Padding|Variable|Padding|Header|....uninitialized Data ....
                //! Size of padding can be zero if data after padding is already aligned.
                MetaData* const m_mem;
#    ifndef NDEBUG
                //! max number of meta data entries
                uint32_t const m_capacity;
#    endif
            };
        } // namespace detail

        class StaticSharedMemory : private detail::PtrLookupTable
        {
        public:
            /** number of bytes required for bookkeeping of maxNumberOfAllocations unique allocations
             *
             * @param maxNumUniqueAllocations number of unique allocation a user is allowed to perform
             * @return bytes required to store lookup meta data
             */
            static consteval uint32_t sizeLookupBufferInBytes(uint32_t maxNumUniqueAllocations)
            {
                return detail::PtrLookupTable::sizeLookupBufferInBytes(maxNumUniqueAllocations);
            }

            StaticSharedMemory(StaticSharedMemory const&) = delete;

            /** Construct shared memory allocator
             * @param accessor local memory accessor to store lookup meta data
             *                 bytes required to store N unique allocation can be calculated with
             * sizeLookupBufferInBytes()
             */
            StaticSharedMemory(sycl::local_accessor<std::byte> const& accessor)
                : PtrLookupTable(
                      reinterpret_cast<std::byte*>(accessor.get_multi_ptr<sycl::access::decorated::no>().get()),
                      static_cast<uint32_t>(accessor.size()))

            {
            }

            using Base = detail::PtrLookupTable;

            template<typename T, size_t T_unique>
            T& allocVar()
            {
                T* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    data = Base::template alloc<T, T_unique>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }
        };

    } // namespace oneApi
} // namespace alpaka::onAcc

#endif
