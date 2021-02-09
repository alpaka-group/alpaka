/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>
#include <alpaka/wait/Traits.hpp>

#include <CL/sycl.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    namespace detail
    {
        template <typename TDev>
        struct EventGenericSyclImpl final
        {
            EventGenericSyclImpl(TDev const& d) : dev{d}
            {}
            EventGenericSyclImpl(EventGenericSyclImpl const&) = delete;
            auto operator=(EventGenericSyclImpl const&) -> EventGenericSyclImpl& = delete;
            EventGenericSyclImpl(EventGenericSyclImpl&&) = default;
            auto operator=(EventGenericSyclImpl&&) -> EventGenericSyclImpl& = default;
            ~EventGenericSyclImpl() = default;

            TDev dev;
            cl::sycl::event event{};
        };
    }
    //#############################################################################
    //! The SYCL device event.
    template <typename TDev>
    class EventGenericSycl final
    {
        friend struct traits::GetDev<EventGenericSycl<TDev>>;
        friend struct traits::IsComplete<EventGenericSycl<TDev>>;
        friend struct traits::Enqueue<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::CurrentThreadWaitFor<EventGenericSycl<TDev>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST EventGenericSycl(TDev const& dev)
        : pimpl{std::make_shared<detail::EventGenericSyclImpl<TDev>>(dev)}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }
        //-----------------------------------------------------------------------------
        EventGenericSycl(EventGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        EventGenericSycl(EventGenericSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventGenericSycl const &) -> EventGenericSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventGenericSycl &&) -> EventGenericSycl & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(EventGenericSycl const & rhs) const -> bool
        {
            return (pimpl->event == rhs.pimpl->event);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(EventGenericSycl const & rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~EventGenericSycl() = default;

    private:
        std::shared_ptr<detail::EventGenericSyclImpl<TDev>> pimpl;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device event device get trait specialization.
        template<typename TDev>
        struct GetDev<EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(EventGenericSycl<TDev> const & event)-> TDev
            {
                return event.pimpl->dev;
            }
        };

        //#############################################################################
        //! The SYCL device event test trait specialization.
        template<typename TDev>
        struct IsComplete<EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isComplete(EventGenericSycl<TDev> const & event) -> bool
            {
                using namespace cl::sycl;
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                const auto status = event.pimpl->event.template get_info<info::event::command_execution_status>();
                return (status == info::event_command_status::complete);
            }
        };

        //#############################################################################
        //! The SYCL queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueGenericSyclNonBlocking<TDev>& queue, EventGenericSycl<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.pimpl->event = queue.m_event;
            }
        };

        //#############################################################################
        //! The SYCL queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueGenericSyclBlocking<TDev>& queue, EventGenericSycl<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.pimpl->event = queue.m_event;
            }
        };

        //#############################################################################
        //! The SYCL device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
        //! If the event is not enqueued to a queue the method returns immediately.
        template<typename TDev>
        struct CurrentThreadWaitFor<EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventGenericSycl<TDev> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.pimpl->event.wait_and_throw();
            }
        };

        //#############################################################################
        //! The SYCL queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueGenericSyclNonBlocking<TDev>& queue, EventGenericSycl<TDev> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                queue.m_dependencies.push_back(event.pimpl->event);
            }
        };

        //#############################################################################
        //! The SYCL queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueGenericSyclBlocking<TDev> & queue, EventGenericSycl<TDev> const & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                queue.m_dependencies.push_back(event.pimpl->event);
            }
        };

        //#############################################################################
        //! The SYCL device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
        template<typename TDev>
        struct WaiterWaitFor<TDev, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(TDev& dev, EventGenericSycl<TDev> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                dev.m_dependencies.push_back(event.pimpl->event);
            }
        };
    }
}

#endif
