/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/CallbackThread.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/wait/Traits.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace alpaka
{
    template<typename TDev>
    struct EventGenericThreads;

    namespace generic::detail
    {
#if BOOST_COMP_CLANG
// avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every
// translation unit [-Werror,-Wweak-vtables]" https://stackoverflow.com/a/29288300
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
        //! The CPU device queue implementation.
        template<typename TDev>
        struct QueueGenericThreadsNonBlockingImpl final : IGenericThreadsQueue<TDev>
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
        {
            explicit QueueGenericThreadsNonBlockingImpl(TDev dev) : m_dev(std::move(dev))
            {
            }
            QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev> const&) = delete;
            auto operator=(QueueGenericThreadsNonBlockingImpl<TDev> const&)
                -> QueueGenericThreadsNonBlockingImpl<TDev>& = delete;

            void enqueue(EventGenericThreads<TDev>& ev) final
            {
                alpaka::enqueue(*this, ev);
            }

            void wait(EventGenericThreads<TDev> const& ev) final
            {
                alpaka::wait(*this, ev);
            }

            TDev const m_dev; //!< The device this queue is bound to.
            core::CallbackThread m_workerThread;
        };
    } // namespace generic::detail

    //! The CPU device queue.
    template<typename TDev>
    struct QueueGenericThreadsNonBlocking final
        : concepts::Implements<ConceptCurrentThreadWaitFor, QueueGenericThreadsNonBlocking<TDev>>
        , concepts::Implements<ConceptQueue, QueueGenericThreadsNonBlocking<TDev>>
        , concepts::Implements<ConceptGetDev, QueueGenericThreadsNonBlocking<TDev>>
    {
        explicit QueueGenericThreadsNonBlocking(TDev const& dev)
            : m_spQueueImpl(std::make_shared<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>>(dev))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            dev.registerQueue(m_spQueueImpl);
        }

        auto operator==(QueueGenericThreadsNonBlocking<TDev> const& rhs) const -> bool
        {
            return (m_spQueueImpl == rhs.m_spQueueImpl);
        }

        auto operator!=(QueueGenericThreadsNonBlocking<TDev> const& rhs) const -> bool
        {
            return !(*this == rhs);
        }

        std::shared_ptr<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>> m_spQueueImpl;
    };

    namespace trait
    {
        //! The CPU non-blocking device queue device type trait specialization.
        template<typename TDev>
        struct DevType<QueueGenericThreadsNonBlocking<TDev>>
        {
            using type = TDev;
        };
        //! The CPU non-blocking device queue device get trait specialization.
        template<typename TDev>
        struct GetDev<QueueGenericThreadsNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto getDev(QueueGenericThreadsNonBlocking<TDev> const& queue) -> TDev
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CPU non-blocking device queue event type trait specialization.
        template<typename TDev>
        struct EventType<QueueGenericThreadsNonBlocking<TDev>>
        {
            using type = EventGenericThreads<TDev>;
        };

        //! The CPU non-blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TDev, typename TTask>
        struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, TTask>
        {
            ALPAKA_FN_HOST static auto enqueue(
                [[maybe_unused]] QueueGenericThreadsNonBlocking<TDev>& queue,
                [[maybe_unused]] TTask const& task) -> void
            {
                queue.m_spQueueImpl->m_workerThread.submit(task);
            }
        };
        //! The CPU non-blocking device queue test trait specialization.
        template<typename TDev>
        struct Empty<QueueGenericThreadsNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto empty(QueueGenericThreadsNonBlocking<TDev> const& queue) -> bool
            {
                return queue.m_spQueueImpl->m_workerThread.taskCount() == 0;
            }
        };
    } // namespace trait
} // namespace alpaka

#include <alpaka/event/EventGenericThreads.hpp>
