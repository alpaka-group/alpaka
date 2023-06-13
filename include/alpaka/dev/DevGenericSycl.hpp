/* Copyright 2022 Jan Stephan, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/pltf/Traits.hpp"
#include "alpaka/queue/Properties.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/queue/sycl/QueueGenericSyclBase.hpp"
#include "alpaka/traits/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <CL/sycl.hpp>

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl;

    namespace detail
    {
        class DevGenericSyclImpl
        {
        public:
            DevGenericSyclImpl(sycl::device device, sycl::context context)
                : m_device{std::move(device)}
                , m_context{std::move(context)}
            {
            }

            // Don't call this without locking first!
            auto clean_queues() -> void
            {
                // Clean up dead queues
                auto const start = std::begin(m_queues);
                auto const old_end = std::end(m_queues);
                auto const new_end = std::remove_if(start, old_end, [](auto q_ptr) { return q_ptr.expired(); });
                m_queues.erase(new_end, old_end);
            }

            auto register_queue(std::shared_ptr<QueueGenericSyclImpl> const& queue) -> void
            {
                std::lock_guard<std::shared_mutex> lock{m_mutex};

                clean_queues();
                m_queues.emplace_back(queue);
            }

            auto register_dependency(sycl::event event) -> void
            {
                std::shared_lock<std::shared_mutex> lock{m_mutex};

                for(auto& q_ptr : m_queues)
                {
                    if(auto ptr = q_ptr.lock(); ptr != nullptr)
                        ptr->register_dependency(event);
                }
            }

            auto wait()
            {
                std::shared_lock<std::shared_mutex> lock{m_mutex};

                for(auto& q_ptr : m_queues)
                {
                    if(auto ptr = q_ptr.lock(); ptr != nullptr)
                        ptr->wait();
                }
            }

            auto get_device() const -> sycl::device
            {
                return m_device;
            }

            auto get_context() const -> sycl::context
            {
                return m_context;
            }

        private:
            sycl::device m_device;
            sycl::context m_context;
            std::vector<std::weak_ptr<QueueGenericSyclImpl>> m_queues;
            std::shared_mutex mutable m_mutex;
        };
    } // namespace detail

    //! The SYCL device handle.
    template<typename TPltf>
    class DevGenericSycl
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevGenericSycl<TPltf>>
        , public concepts::Implements<ConceptDev, DevGenericSycl<TPltf>>
    {
    public:
        DevGenericSycl(sycl::device device, sycl::context context)
            : m_impl{std::make_shared<detail::DevGenericSyclImpl>(std::move(device), std::move(context))}
        {
        }

        friend auto operator==(DevGenericSycl const& lhs, DevGenericSycl const& rhs) -> bool
        {
            return (lhs.m_impl == rhs.m_impl);
        }

        friend auto operator!=(DevGenericSycl const& lhs, DevGenericSycl const& rhs) -> bool
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const -> std::pair<sycl::device, sycl::context>
        {
            return std::make_pair(m_impl->get_device(), m_impl->get_context());
        }

        std::shared_ptr<detail::DevGenericSyclImpl> m_impl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device name get trait specialization.
    template<typename TPltf>
    struct GetName<DevGenericSycl<TPltf>>
    {
        static auto getName(DevGenericSycl<TPltf> const& dev) -> std::string
        {
            auto const device = dev.getNativeHandle().first;
            return device.template get_info<sycl::info::device::name>();
        }
    };

    //! The SYCL device available memory get trait specialization.
    template<typename TPltf>
    struct GetMemBytes<DevGenericSycl<TPltf>>
    {
        static auto getMemBytes(DevGenericSycl<TPltf> const& dev) -> std::size_t
        {
            auto const device = dev.getNativeHandle().first;
            return device.template get_info<sycl::info::device::global_mem_size>();
        }
    };

    //! The SYCL device free memory get trait specialization.
    template<typename TPltf>
    struct GetFreeMemBytes<DevGenericSycl<TPltf>>
    {
        static auto getFreeMemBytes(DevGenericSycl<TPltf> const& /* dev */) -> std::size_t
        {
            static_assert(!sizeof(TPltf), "Querying free device memory not supported for SYCL devices.");
            return std::size_t{};
        }
    };

    //! The SYCL device warp size get trait specialization.
    template<typename TPltf>
    struct GetWarpSizes<DevGenericSycl<TPltf>>
    {
        static auto getWarpSizes(DevGenericSycl<TPltf> const& dev) -> std::vector<std::size_t>
        {
            auto const device = dev.getNativeHandle().first;
            return device.template get_info<sycl::info::device::sub_group_sizes>();
        }
    };

    //! The SYCL device reset trait specialization.
    template<typename TPltf>
    struct Reset<DevGenericSycl<TPltf>>
    {
        static auto reset(DevGenericSycl<TPltf> const&) -> void
        {
            static_assert(!sizeof(TPltf), "Explicit device reset not supported for SYCL devices");
        }
    };

    //! The SYCL device native handle trait specialization.
    template<typename TPltf>
    struct NativeHandle<DevGenericSycl<TPltf>>
    {
        [[nodiscard]] static auto getNativeHandle(DevGenericSycl<TPltf> const& dev)
        {
            return dev.getNativeHandle();
        }
    };

    //! The SYCL device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct BufType<DevGenericSycl<TPltf>, TElem, TDim, TIdx>
    {
        using type = BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>>;
    };

    //! The SYCL device platform type trait specialization.
    template<typename TPltf>
    struct PltfType<DevGenericSycl<TPltf>>
    {
        using type = TPltf;
    };

    //! The thread SYCL device wait specialization.
    template<typename TPltf>
    struct CurrentThreadWaitFor<DevGenericSycl<TPltf>>
    {
        static auto currentThreadWaitFor(DevGenericSycl<TPltf> const& dev) -> void
        {
            dev.m_impl->wait();
        }
    };

    //! The SYCL blocking queue trait specialization.
    template<typename TPltf>
    struct QueueType<DevGenericSycl<TPltf>, Blocking>
    {
        using type = detail::QueueGenericSyclBase<DevGenericSycl<TPltf>, true>;
    };

    //! The SYCL non-blocking queue trait specialization.
    template<typename TPltf>
    struct QueueType<DevGenericSycl<TPltf>, NonBlocking>
    {
        using type = detail::QueueGenericSyclBase<DevGenericSycl<TPltf>, false>;
    };
} // namespace alpaka::trait

#endif
