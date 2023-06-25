/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/fence/Traits.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <CL/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        template<typename TAlpakaMemScope>
        struct SyclFenceProps
        {
        };

        template<>
        struct SyclFenceProps<alpaka::memory_scope::Block>
        {
            static constexpr auto scope = sycl::memory_scope::work_group;
            static constexpr auto space = sycl::access::address_space::local_space;
        };

        template<>
        struct SyclFenceProps<alpaka::memory_scope::Device>
        {
            static constexpr auto scope = sycl::memory_scope::device;
            static constexpr auto space = sycl::access::address_space::global_space;
        };
    } // namespace detail

    //! The SYCL memory fence.
    class MemFenceGenericSycl : public concepts::Implements<ConceptMemFence, MemFenceGenericSycl>
    {
    public:
        MemFenceGenericSycl(
            sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::device> global_dummy,
            sycl::local_accessor<int> local_dummy)
            : m_global_dummy{global_dummy}
            , m_local_dummy{local_dummy}
        {
        }

        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::device> m_global_dummy;
        sycl::local_accessor<int> m_local_dummy;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename TMemScope>
    struct MemFence<MemFenceGenericSycl, TMemScope>
    {
        static auto mem_fence(MemFenceGenericSycl const& fence, TMemScope const&)
        {
            static constexpr auto scope = detail::SyclFenceProps<TMemScope>::scope;
            static constexpr auto space = detail::SyclFenceProps<TMemScope>::space;
            auto dummy
                = (scope == sycl::memory_scope::work_group)
                      ? sycl::atomic_ref<int, sycl::memory_order::relaxed, scope, space>{fence.m_local_dummy[0]}
                      : sycl::atomic_ref<int, sycl::memory_order::relaxed, scope, space>{fence.m_global_dummy[0]};
            auto const dummy_val = dummy.load();
            sycl::atomic_fence(sycl::memory_order::acq_rel, scope);
            dummy.store(dummy_val);
        }
    };
} // namespace alpaka::trait

#endif
