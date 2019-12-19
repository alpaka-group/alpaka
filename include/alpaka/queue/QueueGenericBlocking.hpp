/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <atomic>
#include <mutex>

namespace alpaka
{
    namespace event
    {
        template<typename TDev>
        class EventGeneric;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace generic
        {
            namespace detail
            {
#if BOOST_COMP_CLANG
    // avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit [-Werror,-Wweak-vtables]"
    // https://stackoverflow.com/a/29288300
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif
                //#############################################################################
                //! The CPU device queue implementation.
                template<
                    typename TDev,
                    typename TIFace>
                class QueueGenericBlockingImpl final : public TIFace
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                {
                public:
                    //-----------------------------------------------------------------------------
                    QueueGenericBlockingImpl(
                        TDev const & dev) noexcept :
                            m_dev(dev),
                            m_bCurrentlyExecutingTask(false)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueGenericBlockingImpl(QueueGenericBlockingImpl<TDev, TIFace> const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueGenericBlockingImpl(QueueGenericBlockingImpl<TDev, TIFace> &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericBlockingImpl<TDev, TIFace> const &) -> QueueGenericBlockingImpl<TDev, TIFace> & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericBlockingImpl<TDev, TIFace> &&) -> QueueGenericBlockingImpl<TDev, TIFace> & = delete;

                    //-----------------------------------------------------------------------------
                    void enqueue(event::EventGeneric<TDev> & ev) final
                    {
                        queue::enqueue(*this, ev);
                    }

                    //-----------------------------------------------------------------------------
                    void wait(event::EventGeneric<TDev> const & ev) final
                    {
                        wait::wait(*this, ev);
                    }

                public:
                    TDev const m_dev;            //!< The device this queue is bound to.
                    std::mutex mutable m_mutex;
                    std::atomic<bool> m_bCurrentlyExecutingTask;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        template<
            typename TDev,
            typename TIFace>
        class QueueGenericBlocking final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueGenericBlocking<TDev, TIFace>>
        {
        public:
            //-----------------------------------------------------------------------------
            QueueGenericBlocking(
                TDev const & dev) :
                    m_spQueueImpl(std::make_shared<generic::detail::QueueGenericBlockingImpl<TDev, TIFace>>(dev))
            {
                dev.m_spDevCpuImpl->RegisterQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueGenericBlocking(QueueGenericBlocking<TDev, TIFace> const &) = default;
            //-----------------------------------------------------------------------------
            QueueGenericBlocking(QueueGenericBlocking<TDev, TIFace> &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericBlocking<TDev, TIFace> const &) -> QueueGenericBlocking<TDev, TIFace> & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericBlocking<TDev, TIFace> &&) -> QueueGenericBlocking<TDev, TIFace> & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueGenericBlocking<TDev, TIFace> const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueGenericBlocking<TDev, TIFace> const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueGenericBlocking() = default;

        public:
            std::shared_ptr<generic::detail::QueueGenericBlockingImpl<TDev, TIFace>> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue device type trait specialization.
            template<
                typename TDev,
                typename TIFace>
            struct DevType<
                queue::QueueGenericBlocking<TDev, TIFace>>
            {
                using type = TDev;
            };
            //#############################################################################
            //! The CPU blocking device queue device get trait specialization.
            template<
                typename TDev,
                typename TIFace>
            struct GetDev<
                queue::QueueGenericBlocking<TDev, TIFace>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueGenericBlocking<TDev, TIFace> const & queue)
                -> TDev
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue event type trait specialization.
            template<
                typename TDev,
                typename TIFace>
            struct EventType<
                queue::QueueGenericBlocking<TDev, TIFace>>
            {
                using type = event::EventGeneric<TDev>;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TDev,
                typename TIFace,
                typename TTask>
            struct Enqueue<
                queue::QueueGenericBlocking<TDev, TIFace>,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueGenericBlocking<TDev, TIFace> & queue,
                    TTask const & task)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

                    task();

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
                }
            };
            //#############################################################################
            //! The CPU blocking device queue test trait specialization.
            template<
                typename TDev,
                typename TIFace>
            struct Empty<
                queue::QueueGenericBlocking<TDev, TIFace>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueGenericBlocking<TDev, TIFace> const & queue)
                -> bool
                {
                    return !queue.m_spQueueImpl->m_bCurrentlyExecutingTask;
                }
            };
        }
    }

    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<
                typename TDev,
                typename TIFace>
            struct CurrentThreadWaitFor<
                queue::QueueGenericBlocking<TDev, TIFace>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueGenericBlocking<TDev, TIFace> const & queue)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                }
            };
        }
    }
}

#include <alpaka/event/EventGeneric.hpp>
