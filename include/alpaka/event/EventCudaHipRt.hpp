/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/DevCudaHipRt.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

#include <alpaka/queue/QueueCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/QueueCudaHipRtBlocking.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    namespace event
    {
        namespace cudaHip
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA-HIP RT device event implementation.
                class EventCudaHipImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCudaHipImpl(
                        dev::DevCudaHipRt const & dev,
                        bool bBusyWait) :
                            m_dev(dev),
                            m_cudaHipEvent()

                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // Create the event on the current device with the specified flags. Valid flags include:
                        // - cudaEventDefault: Default event creation flag.
                        // - cudaEventBlockingSync : Specifies that event should use blocking synchronization.
                        //   A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                        // - cudaEventDisableTiming : Specifies that the created event does not need to record timing data.
                        //   Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                        ALPAKA_CUDA_RT_CHECK(
                            cudaEventCreateWithFlags(
                                &m_cudaHipEvent,
                                (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
#else
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        
                        ALPAKA_HIP_RT_CHECK(
                            hipEventCreateWithFlags(
                                &m_cudaHipEvent,
                                (bBusyWait ? hipEventDefault : hipEventBlockingSync) | hipEventDisableTiming));
#endif
                    }
                    //-----------------------------------------------------------------------------
                    EventCudaHipImpl(EventCudaHipImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    EventCudaHipImpl(EventCudaHipImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCudaHipImpl const &) -> EventCudaHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCudaHipImpl &&) -> EventCudaHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventCudaHipImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaEventDestroy required?
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_dev.m_iDevice));
                        // In case event has been recorded but has not yet been completed when cudaEventDestroy() is called, the function will return immediately
                        // and the resources associated with event will be released automatically once the device has completed event.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(cudaEventDestroy(
                            m_cudaHipEvent));
#else
                        ALPAKA_HIP_RT_CHECK(hipSetDevice(
                            m_dev.m_iDevice));
                        ALPAKA_HIP_RT_CHECK(hipEventDestroy(
                            m_cudaHipEvent));
#endif
                    }

                public:
                    dev::DevCudaHipRt const m_dev;   //!< The device this event is bound to.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaEvent_t m_cudaHipEvent;
#else
                    hipEvent_t m_cudaHipEvent;
#endif
                };
            }
        }

        //#############################################################################
        //! The CUDA-HIP RT device event.
        class EventCudaHipRt final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCudaHipRt(
                dev::DevCudaHipRt const & dev,
                bool bBusyWait = true) :
                    m_spEventImpl(std::make_shared<cudaHip::detail::EventCudaHipImpl>(dev, bBusyWait))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            EventCudaHipRt(EventCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            EventCudaHipRt(EventCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCudaHipRt const &) -> EventCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCudaHipRt &&) -> EventCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventCudaHipRt const & rhs) const
            -> bool
            {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return (m_spEventImpl == rhs.m_spEventImpl);
#else
                return (m_spEventImpl->m_cudaHipEvent == rhs.m_spEventImpl->m_cudaHipEvent);
#endif
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCudaHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventCudaHipRt() = default;

        public:
            std::shared_ptr<cudaHip::detail::EventCudaHipImpl> m_spEventImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT device event device get trait specialization.
            template<>
            struct GetDev<
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCudaHipRt const & event)
                -> dev::DevCudaHipRt
                {
                    return event.m_spEventImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT device event test trait specialization.
            template<>
            struct Test<
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventCudaHipRt const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaEventQuery(
                            event.m_spEventImpl->m_cudaHipEvent),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
#else
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipEventQuery(
                            event.m_spEventImpl->m_cudaHipEvent),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
#endif
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaHipRtNonBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtNonBlocking & queue,
                    event::EventCudaHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
#else
                    ALPAKA_HIP_RT_CHECK(hipEventRecord(
#endif
                        event.m_spEventImpl->m_cudaHipEvent,
                        queue.m_spQueueImpl->m_CudaHipQueue));

                }
            };
            //#############################################################################
            //! The CUDA RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaHipRtBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtBlocking & queue,
                    event::EventCudaHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
#else
                    ALPAKA_HIP_RT_CHECK(hipEventRecord(
#endif
                        event.m_spEventImpl->m_cudaHipEvent,
                        queue.m_spQueueImpl->m_CudaHipQueue));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
#else
                    ALPAKA_HIP_RT_CHECK(hipEventSynchronize(
#endif
                        event.m_spEventImpl->m_cudaHipEvent));
                }
            };
            //#############################################################################
            //! The CUDA-HIP RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaHipRtNonBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaHipRtNonBlocking & queue,
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
#endif
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        event.m_spEventImpl->m_cudaHipEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaHipRtBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaHipRtBlocking & queue,
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
#endif
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        event.m_spEventImpl->m_cudaHipEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA-HIP RT device event wait trait specialization.
            //!
            //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
            template<>
            struct WaiterWaitFor<
                dev::DevCudaHipRt,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCudaHipRt & dev,
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
#else
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
#endif
                        

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
#endif
                        nullptr,
                        event.m_spEventImpl->m_cudaHipEvent,
                        0));
                }
            };
        }
    }
}

#endif
