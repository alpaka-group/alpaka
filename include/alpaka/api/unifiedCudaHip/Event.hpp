/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/api/generic.hpp"
#include "alpaka/api/hip/IdxLayer.hpp"
#include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#include "alpaka/api/unifiedCudaHip/MemcpyKind.hpp"
#include "alpaka/api/unifiedCudaHip/concepts.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

#    include "alpaka/core/ApiCudaRt.hpp"

#    include <chrono>
#    include <cstdint>
#    include <sstream>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        template<typename T_Device>
        struct Event : std::enable_shared_from_this<Event<T_Device>>
        {
            using ApiInterface = typename T_Device::ApiInterface;

        public:
            Event(
                internal::concepts::DeviceHandle auto device,
                uint32_t const idx,
                alpaka::concepts::EventPolicyList auto const& policies)
                : m_device(std::move(device))
                , m_idx(idx)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(internal::getNativeHandle(*m_device.get())));

                // Create the event on the current device with the specified flags. Valid flags include:
                // - cuda/hip-EventDefault: Default event creation flag.
                // - cuda/hip-EventBlockingSync : Specifies that event should use blocking synchronization.
                //   A host thread that uses cuda/hip-EventSynchronize() to wait on an event created with this flag
                //   will block until the event actually completes. (currently not used, @todo: check if this is
                //   required, in mainline alpaka this is configurable in the constructor.
                // - cuda/hip-EventDisableTiming : Specifies that the created event does not need to record timing
                // data.
                //   Events created with this flag specified and the cuda/hip-EventBlockingSync flag not specified
                //   will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::eventCreateWithFlags(
                        &m_nativeEvent,
                        policies.getTiming() == timing::enabled
                            ? ApiInterface::eventDefault
                            : ApiInterface::eventDefault | ApiInterface::eventDisableTiming));
            }

            ~Event()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                    ApiInterface,
                    ApiInterface::setDevice(internal::getNativeHandle(*m_device.get())));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                    ApiInterface,
                    ApiInterface::eventSynchronize(getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::eventDestroy(getNativeHandle()));
            }

            Event(Event const&) = delete;
            Event& operator=(Event const&) = delete;

            Event(Event&&) = delete;
            Event& operator=(Event&&) = delete;

            bool operator==(Event const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Event const& other) const
            {
                return !(*this == other);
            }

        private:
            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            typename ApiInterface::Event_t m_nativeEvent;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("unifiedCudaHip::Event id=") + std::to_string(m_idx);
            }

            friend struct onHost::internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_nativeEvent;
            }

            friend struct onHost::internal::Enqueue;

            friend struct onHost::internal::Wait;

            void wait() const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::eventSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Event> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct onHost::internal::IsEventComplete;

            bool isEventComplete()
            {
                typename ApiInterface::Error_t ret = ApiInterface::success;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ApiInterface,
                    ret = ApiInterface::eventQuery(m_nativeEvent),
                    ApiInterface::errorNotReady);
                return (ret == ApiInterface::success);
            }

            friend struct onHost::internal::GetDevice;

            template<typename, typename>
            friend struct onHost::internal::GetElapsedTime::Op;

            friend struct alpaka::internal::GetApi;
        };

    } // namespace unifiedCudaHip

    template<typename T_Device>
    struct internal::GetElapsedTime::Op<unifiedCudaHip::Event<T_Device>, unifiedCudaHip::Event<T_Device>>
    {
        auto operator()(unifiedCudaHip::Event<T_Device>& start, unifiedCudaHip::Event<T_Device>& end) const
            -> std::chrono::duration<double>
        {
            start.wait();
            end.wait();
            auto milliseconds = 0.0F;
            using ApiInterface = typename T_Device::ApiInterface;
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                ApiInterface,
                ApiInterface::eventElapsedTime(&milliseconds, start.getNativeHandle(), end.getNativeHandle()));
            return std::chrono::duration<double, std::milli>{static_cast<double>(milliseconds)};
        }
    };
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::unifiedCudaHip::Event<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal
#endif
