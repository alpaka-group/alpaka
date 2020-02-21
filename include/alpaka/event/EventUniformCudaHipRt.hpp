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

#include <alpaka/dev/DevUniformCudaHipRt.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

#include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    namespace event
    {
        namespace uniform_cuda_hip
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA/HIP RT device event implementation.
                class EventUniformCudaHipImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventUniformCudaHipImpl(
                        dev::DevUniformCudaHipRt const & dev,
                        bool bBusyWait) :
                            m_dev(dev),
                            m_UniformCudaHipEvent()
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
                                &m_UniformCudaHipEvent,
                                (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
#else
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));

                        ALPAKA_HIP_RT_CHECK(
                            hipEventCreateWithFlags(
                                &m_UniformCudaHipEvent,
                                (bBusyWait ? hipEventDefault : hipEventBlockingSync) | hipEventDisableTiming));
#endif
                    }
                    //-----------------------------------------------------------------------------
                    EventUniformCudaHipImpl(EventUniformCudaHipImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    EventUniformCudaHipImpl(EventUniformCudaHipImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventUniformCudaHipImpl const &) -> EventUniformCudaHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventUniformCudaHipImpl &&) -> EventUniformCudaHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventUniformCudaHipImpl()
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
                            m_UniformCudaHipEvent));
#else
                        ALPAKA_HIP_RT_CHECK(hipSetDevice(
                            m_dev.m_iDevice));
                        ALPAKA_HIP_RT_CHECK(hipEventDestroy(
                            m_UniformCudaHipEvent));
#endif
                    }

                public:
                    dev::DevUniformCudaHipRt const m_dev;   //!< The device this event is bound to.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaEvent_t m_UniformCudaHipEvent;
#else
                    hipEvent_t m_UniformCudaHipEvent;
#endif
                };
            }
        }

        //#############################################################################
        //! The CUDA/HIP RT device event.
        class EventUniformCudaHipRt final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, EventUniformCudaHipRt>
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventUniformCudaHipRt(
                dev::DevUniformCudaHipRt const & dev,
                bool bBusyWait = true) :
                    m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventUniformCudaHipImpl>(dev, bBusyWait))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            EventUniformCudaHipRt(EventUniformCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            EventUniformCudaHipRt(EventUniformCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventUniformCudaHipRt const &) -> EventUniformCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventUniformCudaHipRt &&) -> EventUniformCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventUniformCudaHipRt const & rhs) const
            -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventUniformCudaHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventUniformCudaHipRt() = default;

        public:
            std::shared_ptr<uniform_cuda_hip::detail::EventUniformCudaHipImpl> m_spEventImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT device event device get trait specialization.
            template<>
            struct GetDev<
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventUniformCudaHipRt const & event)
                -> dev::DevUniformCudaHipRt
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
            //! The CUDA/HIP RT device event test trait specialization.
            template<>
            struct Test<
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventUniformCudaHipRt const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaEventQuery(
                            event.m_spEventImpl->m_UniformCudaHipEvent),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
#else
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipEventQuery(
                            event.m_spEventImpl->m_UniformCudaHipEvent),
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
            //! The CUDA/HIP RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueUniformCudaHipRtNonBlocking,
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtNonBlocking & queue,
                    event::EventUniformCudaHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
#else
                    ALPAKA_HIP_RT_CHECK(hipEventRecord(
                         event.m_spEventImpl->m_UniformCudaHipEvent,
                         queue.m_spQueueImpl->m_UniformCudaHipQueue));
#endif
                }
            };
            //#############################################################################
            //! The CUDA/HIP RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueUniformCudaHipRtBlocking,
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtBlocking & queue,
                    event::EventUniformCudaHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
#else
                    ALPAKA_HIP_RT_CHECK(hipEventRecord(
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
#endif
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventUniformCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
                        event.m_spEventImpl->m_UniformCudaHipEvent));
#else
                    ALPAKA_HIP_RT_CHECK(hipEventSynchronize(
                        event.m_spEventImpl->m_UniformCudaHipEvent));
#endif
                }
            };
            //#############################################################################
            //! The CUDA/HIP RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueUniformCudaHipRtNonBlocking,
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueUniformCudaHipRtNonBlocking & queue,
                    event::EventUniformCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_UniformCudaHipQueue,
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        0));
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
                        queue.m_spQueueImpl->m_UniformCudaHipQueue,
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        0));
#endif
                }
            };
            //#############################################################################
            //! The CUDA/HIP RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueUniformCudaHipRtBlocking,
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueUniformCudaHipRtBlocking & queue,
                    event::EventUniformCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_UniformCudaHipQueue,
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        0));
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
                        queue.m_spQueueImpl->m_UniformCudaHipQueue,
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        0));
#endif
                }
            };
            //#############################################################################
            //! The CUDA/HIP RT device event wait trait specialization.
            //!
            //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
            template<>
            struct WaiterWaitFor<
                dev::DevUniformCudaHipRt,
                event::EventUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevUniformCudaHipRt & dev,
                    event::EventUniformCudaHipRt const & event)
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
                        nullptr,
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        0));
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamWaitEvent(
                        nullptr,
                        event.m_spEventImpl->m_UniformCudaHipEvent,
                        0));
#endif
                }
            };
        }
    }
}

#endif
