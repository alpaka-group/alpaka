/* Copyright 2019 Benjamin Worpitz, René Widera
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
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace alpaka
{
    namespace event
    {
        class EventCudaHipRt;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace cudaHip
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA-HIP RT blocking queue implementation.
                class QueueCudaHipRtBlockingImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueCudaHipRtBlockingImpl(
                        dev::DevCudaHipRt const & dev) :
                            m_dev(dev),
                            m_CudaHipQueue()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            ALPAKA_CUDA_RT_CHECK(
                            cudaStreamCreateWithFlags(
                                &m_CudaHipQueue,
                                cudaStreamNonBlocking));
#else
                            ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
#endif
                        // - [cuda/hip]StreamDefault: Default queue creation flag.
                        // - [cuda/hip]StreamNonBlocking: Specifies that work running in the created queue may run concurrently with work in queue 0 (the NULL queue),
                        //   and that the created queue should perform no implicit synchronization with queue 0.
                        // Create the queue on the current device.
                        // NOTE: [cuda/hip]StreamNonBlocking is required to match the semantic implemented in the alpaka CPU queue.
                        // It would be too much work to implement implicit default queue synchronization on CPU.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            ALPAKA_CUDA_RT_CHECK(
                            cudaStreamCreateWithFlags(
                                &m_CudaHipQueue,
                                cudaStreamNonBlocking));
#else
                            ALPAKA_HIP_RT_CHECK(
                            hipStreamCreateWithFlags(
                                &m_CudaHipQueue,
                                hipStreamNonBlocking));
#endif

                    }
                    //-----------------------------------------------------------------------------
                    QueueCudaHipRtBlockingImpl(QueueCudaHipRtBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCudaHipRtBlockingImpl(QueueCudaHipRtBlockingImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCudaHipRtBlockingImpl const &) -> QueueCudaHipRtBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCudaHipRtBlockingImpl &&) -> QueueCudaHipRtBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~QueueCudaHipRtBlockingImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before [cuda/hip]StreamDestroy required?

                        // In case the device is still doing work in the queue when [cuda/hip]StreamDestroy() is called, the function will return immediately
                        // and the resources associated with queue will be released automatically once the device has completed all work in queue.
                        // -> No need to synchronize here.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));

                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamDestroy(
                                m_CudaHipQueue));
#else
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamDestroy(
                                m_CudaHipQueue));
#endif
                    }

                public:
                    dev::DevCudaHipRt const m_dev;   //!< The device this queue is bound to.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    cudaStream_t m_CudaHipQueue;
#else
                    hipStream_t m_CudaHipQueue;
#endif
#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    int m_callees = 0;
                    std::mutex m_mutex;
#endif
                };
            }
        }

        //#############################################################################
        //! The CUDA-HIP RT blocking queue.
        class QueueCudaHipRtBlocking final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueCudaHipRtBlocking(
                dev::DevCudaHipRt const & dev) :
                m_spQueueImpl(std::make_shared<cudaHip::detail::QueueCudaHipRtBlockingImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueCudaHipRtBlocking(QueueCudaHipRtBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueCudaHipRtBlocking(QueueCudaHipRtBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCudaHipRtBlocking const &) -> QueueCudaHipRtBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCudaHipRtBlocking &&) -> QueueCudaHipRtBlocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueCudaHipRtBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueCudaHipRtBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCudaHipRtBlocking() = default;

        public:
            std::shared_ptr<cudaHip::detail::QueueCudaHipRtBlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCudaHipRtBlocking>
            {
                using type = dev::DevCudaHipRt;
            };
            //#############################################################################
            //! The CUDA-HIP RT blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCudaHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCudaHipRtBlocking const & queue)
                -> dev::DevCudaHipRt
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
            //! The CUDA-HIP RT blocking queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCudaHipRtBlocking>
            {
                using type = event::EventCudaHipRt;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP RT blocking queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCudaHipRtBlocking,
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
                struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
                {
                    std::mutex m_mutex;
                    std::condition_variable m_event;
                    CallbackState state = CallbackState::enqueued;
                };

                //-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                static void CUDART_CB cudaHipRtCallback(cudaStream_t /*queue*/, cudaError_t /*status*/, void *arg)
#else
                static void HIPRT_CB cudaHipRtCallback(hipStream_t /*queue*/, hipError_t /*status*/, void *arg)
#endif
                {
                    // explicitly copy the shared_ptr so that this method holds the state even when the executing thread has already finished.
                    const auto pCallbackSynchronizationData = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

                    // Notify the executing thread.
                    {
                        std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                        pCallbackSynchronizationData->state = CallbackState::notified;
                    }
                    pCallbackSynchronizationData->m_event.notify_one();

                    // Wait for the executing thread to finish the task if it has not already finished.
                    std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                    if(pCallbackSynchronizationData->state != CallbackState::finished)
                    {
                        pCallbackSynchronizationData->m_event.wait(
                            lock,
                            [pCallbackSynchronizationData](){
                                return pCallbackSynchronizationData->state == CallbackState::finished;
                            }
                        );
                    }
                }

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtBlocking & queue,
                    TTask const & task)
                -> void
                {
#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    {
                        // thread-safe callee incrementing
                        std::lock_guard<std::mutex> guard(queue.m_spQueueImpl->m_mutex);
                        queue.m_spQueueImpl->m_callees += 1;
                    }
#endif
                    auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_CUDA_RT_CHECK(cudaStreamAddCallback(
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        cudaHipRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));
#else
                    ALPAKA_HIP_RT_CHECK(hipStreamAddCallback(
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        cudaHipRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));
#endif
                    // We start a new std::thread which stores the task to be executed.
                    // This circumvents the limitation that it is not possible to call CUDA/HIP methods within the CUDA/HIP callback thread.
                    // The CUDA/HIP thread signals the std::thread when it is ready to execute the task.
                    // The CUDA/HIP thread is waiting for the std::thread to signal that it is finished executing the task
                    // before it executes the next task in the queue (CUDA/HIP stream).
                    std::thread t(
                        [pCallbackSynchronizationData,
                            task
#if BOOST_COMP_HCC // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                         ,&queue
#endif
                        ](){

#if BOOST_COMP_HCC // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                            // thread-safe task execution and callee decrementing
                            std::lock_guard<std::mutex> guard(queue.m_spQueueImpl->m_mutex);
#endif

                            // If the callback has not yet been called, we wait for it.
                            {
                                std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                                if(pCallbackSynchronizationData->state != CallbackState::notified)
                                {
                                    pCallbackSynchronizationData->m_event.wait(
                                        lock,
                                        [pCallbackSynchronizationData](){
                                            return pCallbackSynchronizationData->state == CallbackState::notified;
                                        }
                                    );
                                }

                                task();

                                // Notify the waiting CUDA thread.
                                pCallbackSynchronizationData->state = CallbackState::finished;
                            }
                            pCallbackSynchronizationData->m_event.notify_one();

#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                            queue.m_spQueueImpl->m_callees -= 1;
#endif
                        }
                    );

                    t.join();
                }
            };
            //#############################################################################
            //! The CUDA-HIP RT blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCudaHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCudaHipRtBlocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Query is allowed even for queues on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaStreamQuery(queue.m_spQueueImpl->m_CudaHipQueue),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
#elif BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    return (queue.m_spQueueImpl->m_callees==0);
#else
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipStreamQuery(
                            queue.m_spQueueImpl->m_CudaHipQueue),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
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
            //! The CUDA-HIP RT blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCudaHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCudaHipRtBlocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Sync is allowed even for queues on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaHipQueue));

#elif BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    while(queue.m_spQueueImpl->m_callees>0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10u));
                    }
#else
                    // Sync is allowed even for queues on non current device.
                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaHipQueue));
#endif
                }
            };
        }
    }
}

#endif
