/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/Event.hpp"
#include "alpaka/api/unifiedCudaHip/Queue.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP


#    include <cstdint>
#    include <memory>
#    include <mutex>
#    include <sstream>
#    include <vector>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
            using ApiInterface = typename T_Platform::ApiInterface;

        public:
            Device(internal::concepts::PlatformHandle auto platform, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_properties{internal::getDeviceProperties(*m_platform.get(), m_idx)}
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::setDevice(idx));
            }

            ~Device()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::setDevice(getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::deviceSynchronize());
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::deviceReset());
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
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                // Make sure this device is the current thread device (getNativeHandle returns device index)
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::setDevice(getNativeHandle()));
                // Wait for all work queued on this device to finish
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::deviceSynchronize());
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Device<Device>);
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            DeviceProperties m_properties;
            std::vector<std::weak_ptr<unifiedCudaHip::Queue<Device>>> queues;
            std::vector<std::weak_ptr<unifiedCudaHip::Event<Device>>> events;
            std::mutex m_writeGuard;

            std::shared_ptr<Device> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return m_properties.name;
            }

            friend struct onHost::internal::GetNativeHandle;

            [[nodiscard]] int getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct onHost::internal::MakeQueue;

            Handle<unifiedCudaHip::Queue<Device>> makeQueue(alpaka::concepts::QueuePolicyList auto const& policies)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{m_writeGuard};

                auto newQueue
                    = std::make_shared<unifiedCudaHip::Queue<Device>>(std::move(thisHandle), queues.size(), policies);

                queues.emplace_back(newQueue);
                return newQueue;
            }

            friend struct onHost::internal::MakeEvent;

            Handle<unifiedCudaHip::Event<Device>> makeEvent(alpaka::concepts::EventPolicyList auto const& policies)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{m_writeGuard};
                auto newEvent
                    = std::make_shared<unifiedCudaHip::Event<Device>>(std::move(thisHandle), events.size(), policies);

                events.emplace_back(newEvent);
                return newEvent;
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_platform.get());
            }

            auto getFreeGlobalMemBytes() const
            {
                std::size_t freeGlobalMemBytes(0u);
                std::size_t globalMemCapacityBytes(0u);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::setDevice(getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::memGetInfo(&freeGlobalMemBytes, &globalMemCapacityBytes));
                return freeGlobalMemBytes;
            }

            friend struct onHost::internal::Alloc;
            friend struct onHost::internal::AllocDeferred;
            friend struct onHost::internal::AllocUnified;
            friend struct onHost::internal::AllocMapped;
            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::GetFreeGlobalMemBytes;
            friend struct internal::AdjustThreadSpec;
            friend struct onHost::internal::IsDataAccessible;
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Platform>
    struct GetApi::Op<onHost::unifiedCudaHip::Device<T_Platform>>
    {
        inline constexpr auto operator()(auto&& device) const
        {
            return getApi(device.m_platform);
        }
    };
} // namespace alpaka::internal

namespace alpaka::onHost
{
    namespace internal
    {
        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct Alloc::Op<T_Type, unifiedCudaHip::Device<T_Platform>, T_Extents>
        {
            auto operator()(unifiedCudaHip::Device<T_Platform>& device, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                using ApiInterface = typename T_Platform::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::setDevice(device.getNativeHandle()));

                T_Type* ptr = nullptr;
                auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};

                using Idx = typename T_Extents::value_type;

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::malloc((void**) &ptr, static_cast<std::size_t>(extents.x()) * sizeof(T_Type)));
                }
                else if constexpr(dim == 2u)
                {
                    size_t rowPitchInBytes = 0u;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::mallocPitch(
                            (void**) &ptr,
                            &rowPitchInBytes,
                            static_cast<std::size_t>(extents.x()) * sizeof(T_Type),
                            static_cast<std::size_t>(extents.y())));

                    pitches = alpaka::calculatePitches<T_Type>(extents, static_cast<Idx>(rowPitchInBytes));
                }
                else if constexpr(dim >= 3u)
                {
                    auto const extentsNoXY = pCast<size_t>(extents.eraseBack().eraseBack());
                    typename ApiInterface::Extent_t const extentVal = ApiInterface::makeExtent(
                        static_cast<std::size_t>(extents.x()) * sizeof(T_Type),
                        static_cast<std::size_t>(extents.y()),
                        pCast<std::size_t>(extentsNoXY).product());
                    typename ApiInterface::PitchedPtr_t pitchedPtrVal;
                    pitchedPtrVal.ptr = nullptr;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::malloc3D(&pitchedPtrVal, extentVal));

                    ptr = reinterpret_cast<T_Type*>(pitchedPtrVal.ptr);
                    pitches = alpaka::calculatePitches<T_Type>(extents, static_cast<Idx>(pitchedPtrVal.pitch));
                }

                auto deviceDependency = onHost::Device{device.getSharedPtr()};

                auto deleter = [ptr, deviceIdx = device.getNativeHandle(), deviceDependency]()
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::setDevice(deviceIdx));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::free(toVoidPtr(ptr)));
                };

                /** Each CUDA/HIP allocation is aligned to at least 128 byte but typically to 256byte
                 *
                 * @todo check if this value can be derived from the device properties
                 * @todo validate if memory is always aligned to 256 byte
                 */
                constexpr uint32_t alignment = 128u;

                auto buffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return buffer;
            }
        };

        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct AllocUnified::Op<T_Type, unifiedCudaHip::Device<T_Platform>, T_Extents>
        {
            auto operator()(unifiedCudaHip::Device<T_Platform>& device, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                using ApiInterface = typename T_Platform::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                    ApiInterface,
                    ApiInterface::setDevice(device.getNativeHandle()));

                /** Each CUDA/HIP allocation is aligned to at least 128 byte but typically to 256byte
                 *
                 * @todo check if this value can be derived from the device properties
                 * @todo validate if memory is always aligned to 256 byte
                 */
                constexpr uint32_t alignment = 128u;
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};

                T_Type* ptr = nullptr;
                // HIP is failing if zero byte unified memory is allocated, therefore we do not call the allocation
                // method for HIP
                bool isHipZeroByteAllocation = memSizeInByte == 0 && getApi(device) == api::hip;
                if(!isHipZeroByteAllocation)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::mallocManaged((void**) &ptr, memSizeInByte));
                }

                auto deleter = [ptr, deviceIdx = device.getNativeHandle(), deviceDependency]()
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::setDevice(deviceIdx));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::free(toVoidPtr(ptr)));
                };

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

        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct AllocMapped::Op<T_Type, unifiedCudaHip::Device<T_Platform>, T_Extents>
        {
            auto operator()(unifiedCudaHip::Device<T_Platform>& device, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                using ApiInterface = typename T_Platform::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                    ApiInterface,
                    ApiInterface::setDevice(device.getNativeHandle()));

                /** Each CUDA/HIP allocation is aligned to at least 128 byte but typically to 256byte
                 *
                 * @todo check if this value can be derived from the device properties
                 * @todo validate if memory is always aligned to 256 byte
                 */
                constexpr uint32_t alignment = 128u;
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};

                T_Type* ptr = nullptr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::hostMalloc(
                        (void**) &ptr,
                        memSizeInByte,
                        ApiInterface::hostMallocMapped | ApiInterface::hostMallocPortable));

                auto deleter = [ptr, deviceIdx = device.getNativeHandle(), deviceDependency]()
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::setDevice(deviceIdx));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::hostFree(toVoidPtr(ptr)));
                };

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
        struct IsDataAccessible::FirstPath<unifiedCudaHip::Device<T_Platform>, T_Any>
        {
            bool operator()(unifiedCudaHip::Device<T_Platform>& device, T_Any const& view) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::device);
                using ApiInterface = typename T_Platform::ApiInterface;
                typename ApiInterface::PointerAttr_t ptrAttributes;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::pointerGetAttributes(&ptrAttributes, onHost::data(view)));

                auto deviceHandle = device.getNativeHandle();

                // pointer is owned by the device itself
                if(deviceHandle == ptrAttributes.device)
                    return true;
                if(ptrAttributes.type == ApiInterface::memoryTypeManaged)
                    return true;

                return false;
            }
        };

        template<typename T_Platform>
        struct GetDeviceProperties::Op<unifiedCudaHip::Device<T_Platform>>
        {
            DeviceProperties operator()(unifiedCudaHip::Device<T_Platform> const& device) const
            {
                return device.m_properties;
            }
        };

        template<
            typename T_Platform,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            alpaka::concepts::Vector T_NumFrames,
            alpaka::concepts::Vector T_FrameExtents,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct AdjustThreadSpec::
            Op<unifiedCudaHip::Device<T_Platform>, FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>, T_KernelBundle>
        {
            using FrameSpecType = FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>;

            auto operator()(
                unifiedCudaHip::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const&) const requires alpaka::concepts::CVector<T_FrameExtents>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                auto numThreads = frameSpec.getFrameExtents();

                using ApiType = ALPAKA_TYPEOF(getApi(device));
                using DeviceKindType = ALPAKA_TYPEOF(getDeviceKind(device));
                constexpr auto result = api::util::adjustToLimit<
                    alpaka::onHost::getMaxThreadsPerBlock(ApiType{}, DeviceKindType{}, T_Executor{}),
                    0u,
                    1u>(numThreads);
                return ThreadSpec{frameSpec.getNumFrames(), result, frameSpec.getExecutor()};
            }

            auto operator()(
                unifiedCudaHip::Device<T_Platform> const& device,
                FrameSpecType const& frameSpec,
                T_KernelBundle const&) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::device);
                auto numThreadsPerBlocks = frameSpec.getFrameExtents();
                auto const maxThreadsPerBlock = device.m_properties.maxThreadsPerBlock;

                auto result = api::util::adjustToLimit(numThreadsPerBlocks, maxThreadsPerBlock);
                return ThreadSpec{frameSpec.getNumFrames(), result, frameSpec.getExecutor()};
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

#endif
