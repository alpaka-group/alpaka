/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/Event.hpp"
#include "alpaka/api/host/Queue.hpp"
#include "alpaka/api/host/hwloc/utility.hpp"
#include "alpaka/api/host/sysInfo.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/alignedAlloc.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

#include <cstdint>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
        public:
            Device(internal::concepts::PlatformHandle auto platform, uint32_t const idx, uint32_t cpuGroupIdx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_cpuGroupIdx(cpuGroupIdx)
                , m_properties{internal::getDeviceProperties(*m_platform.get(), m_idx)}
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
            }

            ~Device()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                // wait for all queues before we destroy the device
                wait();
            }

            Device(Device const&) = delete;
            Device& operator=(Device const&) = delete;

            Device(Device&&) = delete;
            Device& operator=(Device&&) = delete;

            bool operator==(Device const& other) const
            {
                return m_idx == other.m_idx;
            }

            bool operator!=(Device const& other) const
            {
                return m_idx != other.m_idx;
            }

            void wait()
            {
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::device);
                // Host device synchronization - wait on all queues associated with this device.
                // IMPORTANT: Do not hold queuesGuard across potentially long waits; copy the list first to avoid to be
                // able to work on the list without a guard.
                std::vector<std::function<void()>> tmpQueues;
                {
                    std::lock_guard<std::mutex> lk{m_queuesGuard};
                    tmpQueues = m_queueWaitFns; // copy weak_ptr list
                }
                for(auto& waitFn : tmpQueues)
                {
                    waitFn();
                }
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Device<Device>);
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            uint32_t m_cpuGroupIdx = internal::hwloc::allDomains;
            DeviceProperties m_properties;
            /* Wait function to any created queue
             * The function should capture the queue only as weak pointer else a queue will never be deleted because
             * the queue would hold a shared pointer to the device and the device via the wait function a shared
             * pointer to the queue.
             */
            std::vector<std::function<void()>> m_queueWaitFns;
            std::vector<std::weak_ptr<cpu::Event<Device>>> m_events;
            std::mutex m_queuesGuard;

            std::shared_ptr<Device> getSharedPtr()
            {
                return this->shared_from_this();
            }

            template<typename T_Device>
            friend struct Queue;

            void setThreadAffinity() const
            {
                internal::hwloc::setThreadAffinity(m_cpuGroupIdx);
            }

            template<typename T, alpaka::concepts::MemoryProperty T_Property>
            void pinPointer(T* const ptr, size_t bytes, T_Property property)
            {
                internal::hwloc::pinPointer(ptr, bytes, property, m_cpuGroupIdx);
            }

            bool isNumaAware() const
            {
                return m_cpuGroupIdx != internal::hwloc::allDomains;
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return m_properties.name;
            }

            friend struct internal::GetNativeHandle;

            [[nodiscard]] uint32_t getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct internal::MakeQueue;

            Handle<cpu::Queue<Device>> makeQueue(alpaka::concepts::QueuePolicyList auto const& policies)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{m_queuesGuard};

                auto newQueue = std::make_shared<cpu::Queue<Device>>(
                    std::move(thisHandle),
                    m_queueWaitFns.size(),
                    m_cpuGroupIdx,
                    policies);

                std::weak_ptr<cpu::Queue<Device>> weakPtrToQueue = newQueue;
                m_queueWaitFns.emplace_back(
                    [weakPtrToQueue]
                    {
                        if(auto queue = weakPtrToQueue.lock())
                            internal::wait(*queue);
                    });
                return newQueue;
            }

            friend struct internal::MakeEvent;

            Handle<cpu::Event<Device>> makeEvent(alpaka::concepts::EventPolicyList auto const& policies)
            {
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::event);
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{m_queuesGuard};
                auto newEvent = std::make_shared<cpu::Event<Device>>(std::move(thisHandle), m_events.size(), policies);

                m_events.emplace_back(newEvent);
                return newEvent;
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_platform.get());
            }

            auto getFreeGlobalMemBytes() const
            {
#if ALPAKA_HAS_HWLOC
                if(isNumaAware())
                    return internal::hwloc::getFreeGlobalMemBytes(m_cpuGroupIdx);
#endif
                return onHost::getFreeGlobalMemBytes();
            }

            friend struct internal::Alloc;
            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::GetFreeGlobalMemBytes;
            friend struct internal::AdjustThreadSpec;
            friend struct internal::AllocDeferred;
            friend struct internal::AllocUnified;
            friend struct internal::AllocMapped;
        };
    } // namespace cpu

    namespace trait

    {
        template<typename T_Platform>
        struct IsExecutorSupportedBy::Op<exec::CpuSerial, cpu::Device<T_Platform>> : std::true_type
        {
        };
#if ALPAKA_OMP
        template<typename T_Platform>
        struct IsExecutorSupportedBy::Op<exec::CpuOmpBlocks, cpu::Device<T_Platform>> : std::true_type
        {
        };
#endif
#if ALPAKA_TBB
        template<typename T_Platform>
        struct IsExecutorSupportedBy::Op<exec::CpuTbbBlocks, cpu::Device<T_Platform>> : std::true_type
        {
        };
#endif

        /** The CPU backend places host memory on NUMA nodes, therefore it implements all placement preferences.
         *
         * Without hwloc the placement is a no-op, the properties stay valid so that a single source keeps
         * compiling with and without hwloc.
         */
        template<alpaka::concepts::MemoryProperty T_Property, typename T_Platform>
        requires(
            std::same_as<T_Property, memoryProperty::BestLatency>
            || std::same_as<T_Property, memoryProperty::BestBandwidth>
            || std::same_as<T_Property, memoryProperty::Locality>)
        struct IsMemoryPropertySupportedBy::Op<T_Property, cpu::Device<T_Platform>> : std::true_type
        {
        };
    } // namespace trait

    namespace internal
    {
        template<
            typename T_Type,
            typename T_Platform,
            alpaka::concepts::Vector T_Extents,
            alpaka::concepts::MemoryProperty T_Property>
        struct Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents, T_Property>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents, T_Property property) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};

                T_Type* ptr = reinterpret_cast<T_Type*>(alpaka::core::alignedAlloc(alignment, memSizeInByte));
                device.pinPointer(ptr, memSizeInByte, property);
                // deviceDependency is captured to keep the device alive until the memory is deleted
                auto deleter = [ptr, deviceDependency]() { alpaka::core::alignedFree(alignment, ptr); };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};

                ALPAKA_LOG_INFO(
                    onHost::logger::memory + onHost::logger::device,
                    [&]()
                    {
                        std::stringstream ss;
                        ss << sharedBuffer;
                        return ss.str();
                    });
                return sharedBuffer;
            }
        };

        template<
            typename T_Type,
            typename T_Platform,
            alpaka::concepts::Vector T_Extents,
            alpaka::concepts::MemoryProperty T_Property>
        struct AllocUnified::Op<T_Type, cpu::Device<T_Platform>, T_Extents, T_Property>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents, T_Property property) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                return Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents, T_Property>{}(device, extents, property);
            }
        };

        template<
            typename T_Type,
            typename T_Platform,
            alpaka::concepts::Vector T_Extents,
            alpaka::concepts::MemoryProperty T_Property>
        struct AllocMapped::Op<T_Type, cpu::Device<T_Platform>, T_Extents, T_Property>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents, T_Property property) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                return Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents, T_Property>{}(device, extents, property);
            }
        };

        template<typename T_Platform, typename T_Any>
        struct IsDataAccessible::FirstPath<cpu::Device<T_Platform>, T_Any>
        {
            bool operator()(cpu::Device<T_Platform>& device, T_Any const& view) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                if constexpr(
                    ALPAKA_TYPEOF(getApi(view)){} == api::host
                    && (ALPAKA_TYPEOF(getDeviceKind(device)){} == deviceKind::cpu
                        || ALPAKA_TYPEOF(getDeviceKind(device)){} == deviceKind::numaCpu))
                    return true;
                else
                    return false;
            }
        };

        /** Set number of thread blocks and threads per block to one
         *
         * There is no need to emulate blocks if we have only one thread.
         */
        template<
            typename T_Platform,
            alpaka::concepts::Vector T_NumFrames,
            alpaka::concepts::Vector T_FrameExtents,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct AdjustThreadSpec::
            Op<cpu::Device<T_Platform>, FrameSpec<T_NumFrames, T_FrameExtents, exec::CpuSerial>, T_KernelBundle>
        {
            using FrameSpecType = FrameSpec<T_NumFrames, T_FrameExtents, exec::CpuSerial>;

            auto operator()(
                cpu::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_FrameExtents>
            {
                alpaka::unused(device, kernelBundle);
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel);

                /// @todo add shortcut to create a CVec with equal values
                auto const allOne = ALPAKA_TYPEOF(
                    iotaCVec<typename T_FrameExtents::value_type, T_FrameExtents::dim()>())::template fill<1u>();
                return ThreadSpec{allOne, allOne, frameSpec.getExecutor()};
            }

            auto operator()(
                cpu::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                alpaka::unused(device, kernelBundle);
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel);
                /// @todo add shortcut to create a CVec with equal values
                auto const allOne = ALPAKA_TYPEOF(
                    iotaCVec<typename T_FrameExtents::value_type, T_FrameExtents::dim()>())::template fill<1u>();
                return ThreadSpec{allOne, allOne, frameSpec.getExecutor()};
            }
        };

        template<
            typename T_Platform,
            alpaka::concepts::Executor T_Executor,
            alpaka::concepts::Vector T_NumFrames,
            alpaka::concepts::Vector T_FrameExtents,
            alpaka::concepts::KernelBundle T_KernelBundle>
        requires exec::isSeqExecutor_v<T_Executor>
        struct AdjustThreadSpec::
            Op<cpu::Device<T_Platform>, FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>, T_KernelBundle>
        {
            using FrameSpecType = FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>;

            auto operator()(
                cpu::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_FrameExtents>
            {
                alpaka::unused(device, kernelBundle);
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel);

                // map the number of frames to thread blocks
                auto numThreadBlocks = frameSpec.getNumFrames();
                return ThreadSpec{numThreadBlocks, T_FrameExtents::template fill<1u>(), frameSpec.getExecutor()};
            }

            auto operator()(
                cpu::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                alpaka::unused(device, kernelBundle);
                ALPAKA_LOG_FUNCTION(alpaka::onHost::logger::kernel);

                // map the number of frames to thread blocks
                auto numThreadBlocks = frameSpec.getNumFrames();
                auto const numThreads = Vec<typename T_FrameExtents::value_type, T_FrameExtents::dim()>::fill(1);
                return ThreadSpec{numThreadBlocks, numThreads, frameSpec.getExecutor()};
            }
        };

        template<typename T_Platform>
        struct GetDeviceProperties::Op<cpu::Device<T_Platform>>
        {
            DeviceProperties operator()(cpu::Device<T_Platform> const& device) const
            {
                return device.m_properties;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Platform>
    struct GetApi::Op<onHost::cpu::Device<T_Platform>>
    {
        inline constexpr auto operator()(auto&& device) const
        {
            return alpaka::getApi(device.m_platform);
        }
    };
} // namespace alpaka::internal
