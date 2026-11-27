/* Copyright 2025 Simeon Ehrig, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Queue.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/syclGeneric/Event.hpp"
#include "alpaka/api/syclGeneric/Queue.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
        public:
            Device(internal::concepts::PlatformHandle auto platform, auto const& dev, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_sycl_dev(dev)
                , m_properties{internal::getDeviceProperties(*m_platform.get(), m_idx)}
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
            }

            ~Device()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
            }

            Device(Device const&) = delete;
            Device& operator=(Device const&) = delete;

            Device(Device&&) = delete;
            Device& operator=(Device&&) = delete;

            auto getName() const
            {
                return m_sycl_dev.get_info<sycl::info::device::name>();
            }

            std::shared_ptr<Device<T_Platform>> getSharedPtr()
            {
                return this->shared_from_this();
            }

            [[nodiscard]] Handle<syclGeneric::Queue<Device>> makeQueue(
                alpaka::concepts::QueuePolicyList auto const& policies)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue + onHost::logger::device);
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{m_writeGuard};

                auto newQueue
                    = std::make_shared<syclGeneric::Queue<Device>>(std::move(thisHandle), queues.size(), policies);

                queues.emplace_back(newQueue);
                return newQueue;
            }

            [[nodiscard]] std::pair<sycl::device, sycl::context> getNativeHandle() const noexcept
            {
                return {m_sycl_dev, m_platform->getContext()};
            }

            void wait()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                // Copy queue weak refs under lock then release to avoid blocking other operations while waiting.
                std::vector<std::weak_ptr<syclGeneric::Queue<Device>>> tmpQueues;
                {
                    std::lock_guard<std::mutex> lk{m_writeGuard};
                    tmpQueues = queues;
                }
                for(auto& weakQueue : tmpQueues)
                {
                    if(auto queue = weakQueue.lock())
                    {
                        queue->wait();
                    }
                }
            }

        private:
            friend struct internal::MakeEvent;

            Handle<syclGeneric::Event<Device>> makeEvent(alpaka::concepts::EventPolicyList auto const& policies)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::device);
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{m_writeGuard};
                auto newEvent
                    = std::make_shared<syclGeneric::Event<Device>>(std::move(thisHandle), events.size(), policies);

                events.emplace_back(newEvent);
                return newEvent;
            }

            void _()
            {
                static_assert(internal::concepts::Device<Device>);
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_platform.get());
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            sycl::device m_sycl_dev;

            std::vector<std::weak_ptr<syclGeneric::Queue<Device>>> queues;
            std::vector<std::weak_ptr<syclGeneric::Event<Device>>> events;
            std::mutex m_writeGuard;

            DeviceProperties m_properties;

            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::GetFreeGlobalMemBytes;
            friend struct internal::AdjustThreadSpec;
            friend struct onHost::internal::AllocDeferred;
            friend struct onHost::internal::AllocUnified;
            friend struct onHost::internal::AllocMapped;
            friend struct onHost::internal::IsDataAccessible;
        };
    } // namespace syclGeneric

    namespace internal
    {

        template<
            typename T_Type,
            typename T_Platform,
            alpaka::concepts::Vector T_Extents,
            alpaka::concepts::MemoryProperty T_Property>
        struct Alloc::Op<T_Type, syclGeneric::Device<T_Platform>, T_Extents, T_Property>
        {
            auto operator()(
                syclGeneric::Device<T_Platform>& device,
                T_Extents const& extents,
                [[maybe_unused]] T_Property property) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};
                auto [sycl_device, sycl_context] = device.getNativeHandle();

                T_Type* ptr = reinterpret_cast<T_Type*>(
                    sycl::aligned_alloc_device(alignment, memSizeInByte, sycl_device, sycl_context));
                auto deleter = [ctx = sycl_context, ptr]() { sycl::free(toVoidPtr(ptr), ctx); };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return sharedBuffer;
            }
        };

        template<
            typename T_Type,
            typename T_Platform,
            alpaka::concepts::Vector T_Extents,
            alpaka::concepts::MemoryProperty T_Property>
        struct AllocUnified::Op<T_Type, syclGeneric::Device<T_Platform>, T_Extents, T_Property>
        {
            auto operator()(
                syclGeneric::Device<T_Platform>& device,
                T_Extents const& extents,
                [[maybe_unused]] T_Property property) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};
                auto [sycl_device, sycl_context] = device.getNativeHandle();

                bool isManagedMemorySupported = sycl_device.has(sycl::aspect::usm_shared_allocations);
                if(!isManagedMemorySupported)
                {
                    throw std::runtime_error("Sycl device does not support unified memory allocations.");
                }

                T_Type* ptr = reinterpret_cast<T_Type*>(
                    sycl::aligned_alloc_shared(alignment, memSizeInByte, sycl_device, sycl_context));
                auto deleter = [ctx = sycl_context, ptr]() { sycl::free(toVoidPtr(ptr), ctx); };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return sharedBuffer;
            }
        };

        template<
            typename T_Type,
            typename T_Platform,
            alpaka::concepts::Vector T_Extents,
            alpaka::concepts::MemoryProperty T_Property>
        struct AllocMapped::Op<T_Type, syclGeneric::Device<T_Platform>, T_Extents, T_Property>
        {
            auto operator()(
                syclGeneric::Device<T_Platform>& device,
                T_Extents const& extents,
                [[maybe_unused]] T_Property property) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};
                auto [_, sycl_context] = device.getNativeHandle();

                T_Type* ptr
                    = reinterpret_cast<T_Type*>(sycl::aligned_alloc_host(alignment, memSizeInByte, sycl_context));
                auto deleter = [ctx = sycl_context, ptr]() { sycl::free(toVoidPtr(ptr), ctx); };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return sharedBuffer;
            }
        };

        template<typename T_Platform, typename T_Any>
        struct IsDataAccessible::FirstPath<syclGeneric::Device<T_Platform>, T_Any>
        {
            bool operator()(syclGeneric::Device<T_Platform>& device, T_Any const& view) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                auto [sycl_device, sycl_context] = device.getNativeHandle();
                auto sycl_alloc_type = sycl::get_pointer_type(data(view), sycl_context);

                if(sycl_alloc_type != sycl::usm::alloc::unknown)
                {
                    try
                    {
                        sycl::device deviceAssociatedWithData = sycl::get_pointer_device(data(view), sycl_context);
                        if(deviceAssociatedWithData == sycl_device)
                        {
                            // sycl device allocated the memory
                            return true;
                        }
                    }
                    catch(...)
                    {
                    }
                }

                if(sycl_alloc_type == sycl::usm::alloc::shared)
                {
                    // is shared within the device context
                    return true;
                }
                else if(sycl_alloc_type == sycl::usm::alloc::unknown)
                {
                    // assume that a sycl cpu device can always access host memory
                    if constexpr(
                        ALPAKA_TYPEOF(getApi(view)){} == api::host
                        && (ALPAKA_TYPEOF(getDeviceKind(device)){} == deviceKind::cpu
                            || ALPAKA_TYPEOF(getDeviceKind(device)){} == deviceKind::numaCpu))
                        return true;
                }

                return false;
            }
        };

        template<typename T_Platform>
        struct GetDeviceProperties::Op<syclGeneric::Device<T_Platform>>
        {
            DeviceProperties operator()(syclGeneric::Device<T_Platform> const& device) const
            {
                return device.m_properties;
            }
        };

        template<
            typename T_Platform,
            alpaka::concepts::Executor T_Executor,
            alpaka::concepts::Vector T_NumFrames,
            alpaka::concepts::Vector T_FrameExtents,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct AdjustThreadSpec::
            Op<syclGeneric::Device<T_Platform>, FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>, T_KernelBundle>
        {
            using FrameSpecType = FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>;

            auto operator()(
                syclGeneric::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_FrameExtents>
            {
                alpaka::unused(device, kernelBundle);
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::device);
                auto numThreads = frameSpec.getFrameExtents();

                using ApiType = ALPAKA_TYPEOF(getApi(device));
                using DeviceKindType = ALPAKA_TYPEOF(getDeviceKind(device));
                constexpr auto result = api::util::adjustToLimit<
                    alpaka::onHost::getMaxThreadsPerBlock(ApiType{}, DeviceKindType{}, T_Executor{}),
                    0u,
                    1u>(numThreads);
                return ThreadSpec{frameSpec.getNumFrames(), result};
            }

            auto operator()(
                syclGeneric::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                alpaka::unused(kernelBundle);
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::device);
                auto numThreadsPerBlocks = frameSpec.getFrameExtents();
                auto const maxThreadsPerBlock = device.m_properties.maxThreadsPerBlock;

                auto result = api::util::adjustToLimit(numThreadsPerBlocks, maxThreadsPerBlock);
                return ThreadSpec{frameSpec.getNumFrames(), result};
            }
        };

    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Platform>
    struct GetApi::Op<onHost::syclGeneric::Device<T_Platform>>
    {
        decltype(auto) operator()(auto&& device) const
        {
            return internal::getApi(*device.m_platform.get());
        }
    };
} // namespace alpaka::internal

#endif
