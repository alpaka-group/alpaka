/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/dyn/Traits.hpp"

#include <cstddef>

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    include <CL/sycl.hpp>

namespace alpaka
{
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynGenericSycl
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

        BlockSharedMemDynGenericSycl(sycl::local_accessor<std::byte> accessor) : m_accessor{accessor}
        {
        }

        sycl::local_accessor<std::byte> m_accessor;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename T>
    struct GetDynSharedMem<T, BlockSharedMemDynGenericSycl>
    {
        static auto getMem(BlockSharedMemDynGenericSycl const& shared) -> T*
        {
            auto void_ptr = sycl::multi_ptr<void, sycl::access::address_space::local_space>{shared.m_accessor};
            auto t_ptr = static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space>>(void_ptr);
            return t_ptr.get();
        }
    };
} // namespace alpaka::trait

#endif
