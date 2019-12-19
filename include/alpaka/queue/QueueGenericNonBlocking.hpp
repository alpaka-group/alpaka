/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/queue/IGenericQueue.hpp>

#include <type_traits>
#include <thread>
#include <mutex>
#include <future>

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
                    typename TDev>
                class QueueGenericNonBlockingImpl final : public IGenericQueue<TDev>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                {
                private:
                    //#############################################################################
                    using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
                        std::size_t,
                        std::thread,                // The concurrent execution type.
                        std::promise,               // The promise type.
                        void,                       // The type yielding the current concurrent execution.
                        std::mutex,                 // The mutex type to use. Only required if TisYielding is true.
                        std::condition_variable,    // The condition variable type to use. Only required if TisYielding is true.
                        false>;                     // If the threads should yield.

                public:
                    //-----------------------------------------------------------------------------
                    QueueGenericNonBlockingImpl(
                        TDev const & dev) :
                            m_dev(dev),
                            m_workerThread(1u)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueGenericNonBlockingImpl(QueueGenericNonBlockingImpl<TDev> const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueGenericNonBlockingImpl(QueueGenericNonBlockingImpl<TDev> &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericNonBlockingImpl<TDev> const &) -> QueueGenericNonBlockingImpl<TDev> & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericNonBlockingImpl<TDev> &&) -> QueueGenericNonBlockingImpl<TDev> & = delete;

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

                    ThreadPool m_workerThread;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        template<
            typename TDev>
        class QueueGenericNonBlocking final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueGenericNonBlocking<TDev>>
        {
        public:
            //-----------------------------------------------------------------------------
            QueueGenericNonBlocking(
                TDev const & dev) :
                    m_spQueueImpl(std::make_shared<generic::detail::QueueGenericNonBlockingImpl<TDev>>(dev))
            {
                dev.m_spDevCpuImpl->RegisterQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueGenericNonBlocking(QueueGenericNonBlocking<TDev> const &) = default;
            //-----------------------------------------------------------------------------
            QueueGenericNonBlocking(QueueGenericNonBlocking<TDev> &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericNonBlocking<TDev> const &) -> QueueGenericNonBlocking<TDev> & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericNonBlocking<TDev> &&) -> QueueGenericNonBlocking<TDev> & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueGenericNonBlocking<TDev> const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueGenericNonBlocking<TDev> const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueGenericNonBlocking() = default;

        public:
            std::shared_ptr<generic::detail::QueueGenericNonBlockingImpl<TDev>> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU non-blocking device queue device type trait specialization.
            template<
                typename TDev>
            struct DevType<
                queue::QueueGenericNonBlocking<TDev>>
            {
                using type = TDev;
            };
            //#############################################################################
            //! The CPU non-blocking device queue device get trait specialization.
            template<
                typename TDev>
            struct GetDev<
                queue::QueueGenericNonBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueGenericNonBlocking<TDev> const & queue)
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
            //! The CPU non-blocking device queue event type trait specialization.
            template<
                typename TDev>
            struct EventType<
                queue::QueueGenericNonBlocking<TDev>>
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
            //! The CPU non-blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TDev,
                typename TTask>
            struct Enqueue<
                queue::QueueGenericNonBlocking<TDev>,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue::QueueGenericNonBlocking<TDev> & queue,
                    TTask const & task)
#else
                    queue::QueueGenericNonBlocking<TDev> &,
                    TTask const &)
#endif
                -> void
                {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue.m_spQueueImpl->m_workerThread.enqueueTask(
                        task);
#endif
                }
            };
            //#############################################################################
            //! The CPU non-blocking device queue test trait specialization.
            template<
                typename TDev>
            struct Empty<
                queue::QueueGenericNonBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueGenericNonBlocking<TDev> const & queue)
                -> bool
                {
                    return queue.m_spQueueImpl->m_workerThread.isIdle();
                }
            };
        }
    }
}

#include <alpaka/event/EventGeneric.hpp>
