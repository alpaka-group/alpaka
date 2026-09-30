/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/api/host/Api.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/EventPolicyList.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Device>
        struct Event : std::enable_shared_from_this<Event<T_Device>>
        {
        public:
            Event(
                internal::concepts::DeviceHandle auto device,
                uint32_t const idx,
                alpaka::concepts::EventPolicyList auto const& policies)
                : m_device(std::move(device))
                , m_idx(idx)
                , m_timingEnabled(policies.getTiming() == timing::enabled)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
            }

            ~Event()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                try
                {
                    internal::wait(*this);
                }
                catch(...)
                {
                    std::cerr << "Waiting for the host event failed during destruction, e.g. a callback task "
                                 "threw."
                              << std::endl;
                }
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

            //!< The mutex used to synchronize access to the event.
            std::mutex mutable m_mutex;
            //!< The future signaling the event completion.
            std::shared_future<void> m_future;
            //!< The number of times this event has been enqueued.
            std::size_t m_enqueueCount = 0u;
            //!< The time this event has been ready the last time.
            //!< Ready means that the event was not waiting within a queue
            //!< (not enqueued or already completed). If m_enqueueCount ==
            //!< m_LastReadyEnqueueCount, the event is currently not enqueued
            std::size_t m_lastReadyEnqueueCount = 0u;
            bool m_timingEnabled = false;
            std::chrono::steady_clock::time_point m_timestamp;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::Event id=") + std::to_string(m_idx);
            }

            friend struct internal::GetNativeHandle;
            friend struct internal::Enqueue;
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

            friend struct onHost::internal::GetDevice;

            friend struct internal::IsEventComplete;

            /** Check if the event is ready.
             *
             * @attention Do not call this method without holding the event lock.
             *
             * @return true if the event is ready, false otherwise
             */
            bool isReady() noexcept
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                return (m_lastReadyEnqueueCount == m_enqueueCount);
            }

            /** Check if the event is complete.
             *
             * @attention Should not be called if the event lock is acquired, because it could lead to a deadlock.
             *
             * @return true if the event is complete, false otherwise
             */
            bool isEventComplete() noexcept
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                std::lock_guard<std::mutex> lk(m_mutex);
                return isReady();
            }

            friend struct internal::WaitFor;
            friend struct internal::Wait;
            template<typename, typename>
            friend struct internal::GetElapsedTime::Op;

            void wait()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event);
                std::unique_lock<std::mutex> lk(m_mutex);
                size_t enqueueCount = m_enqueueCount;

                while(enqueueCount > m_lastReadyEnqueueCount)
                {
                    auto future = m_future;
                    lk.unlock();
                    future.get();
                    lk.lock();
                }
            }

            friend struct alpaka::internal::GetApi;
        };

    } // namespace cpu

    template<typename T_Device>
    struct internal::GetElapsedTime::Op<cpu::Event<T_Device>, cpu::Event<T_Device>>
    {
        auto operator()(cpu::Event<T_Device>& start, cpu::Event<T_Device>& end) const -> std::chrono::duration<double>
        {
            start.wait();
            end.wait();
            if(&start == &end)
            {
                std::lock_guard<std::mutex> lock{start.m_mutex};
                if(start.m_enqueueCount == 0u)
                    throw std::logic_error{"Elapsed time requires recorded events"};
                if(!start.m_timingEnabled)
                    throw std::logic_error{"Elapsed time requires timing-enabled events"};
                return std::chrono::duration<double>::zero();
            }

            std::scoped_lock lock{start.m_mutex, end.m_mutex};
            if(start.m_enqueueCount == 0u || end.m_enqueueCount == 0u)
                throw std::logic_error{"Elapsed time requires recorded events"};
            if(!start.m_timingEnabled || !end.m_timingEnabled)
                throw std::logic_error{"Elapsed time requires timing-enabled events"};
            return end.m_timestamp - start.m_timestamp;
        }
    };
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::Event<T_Device>>
    {
        inline constexpr auto operator()(auto&& event) const
        {
            return alpaka::getApi(event.m_device);
        }
    };
} // namespace alpaka::internal
