/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include "alpaka/block/shared/dyn/Traits.hpp"

#    include <CL/sycl.hpp>

#    include <cstddef>

namespace alpaka
{
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynGenericSycl
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

        BlockSharedMemDynGenericSycl(
            sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> accessor)
            : m_accessor{accessor}
        {
        }

        sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> m_accessor;
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
