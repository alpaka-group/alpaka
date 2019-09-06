/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/dev/DevOmp4.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Omp4.hpp>

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

// namespace alpaka
// {
//     namespace event
//     {
//         class EventOmp4;
//     }
// }

namespace alpaka
{
    namespace queue
    {
        namespace omp4
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT blocking queue implementation.
                class QueueOmp4BlockingImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueOmp4BlockingImpl(
                        dev::DevOmp4 const & dev) noexcept :
                            m_dev(dev)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    QueueOmp4BlockingImpl(QueueOmp4BlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueOmp4BlockingImpl(QueueOmp4BlockingImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueOmp4BlockingImpl const &) -> QueueOmp4BlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueOmp4BlockingImpl &&) -> QueueOmp4BlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~QueueOmp4BlockingImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    }

                public:
                    dev::DevOmp4 const m_dev;   //!< The device this queue is bound to.
                };
            }
        }

        //#############################################################################
        //! The CUDA RT blocking queue.
        class QueueOmp4Blocking final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueOmp4Blocking(
                dev::DevOmp4 const & dev) :
                m_spQueueImpl(std::make_shared<omp4::detail::QueueOmp4BlockingImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueOmp4Blocking(QueueOmp4Blocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueOmp4Blocking(QueueOmp4Blocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueOmp4Blocking const &) -> QueueOmp4Blocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueOmp4Blocking &&) -> QueueOmp4Blocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueOmp4Blocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueOmp4Blocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueOmp4Blocking() = default;

        public:
            std::shared_ptr<omp4::detail::QueueOmp4BlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueOmp4Blocking>
            {
                using type = dev::DevOmp4;
            };
            //#############################################################################
            //! The CUDA RT blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueOmp4Blocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueOmp4Blocking const & queue)
                -> dev::DevOmp4
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
            //! The CUDA RT blocking queue event type trait specialization.
            // template<>
            // struct EventType<
            //     queue::QueueOmp4Blocking>
            // {
            //     using type = event::EventOmp4;
            // };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT blocking queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                alpaka::queue::QueueOmp4Blocking,
                TTask>
            {
                //#############################################################################
                enum class CallbackState
                {
                    enqueued,
                    notified,
                    finished,
                };

                //#############################################################################

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueOmp4Blocking & queue,
                    TTask const & task)
                -> void
                {
                    alpaka::ignore_unused(queue); //! \TODO
                    task();
                }
            };
            //#############################################################################
            //! The CUDA RT blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueOmp4Blocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueOmp4Blocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    alpaka::ignore_unused(queue); //! \TODO

                    return true;
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueOmp4Blocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueOmp4Blocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    alpaka::ignore_unused(queue); //! \TODO
#pragma omp taskwait 
                }
            };
        }
    }
}

#endif
