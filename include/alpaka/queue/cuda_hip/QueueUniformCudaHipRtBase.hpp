/* Copyright 2022 Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#if !defined(ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE)
#    error This is an internal header file, and should never be included directly.
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#error This file should not be included with ALPAKA_ACC_GPU_CUDA_ENABLED and ALPAKA_ACC_GPU_HIP_ENABLED both defined.
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(alpaka_queue_cuda_hip_QueueUniformCudaHipRtBase_hpp_CUDA)        \
    || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(alpaka_queue_cuda_hip_QueueUniformCudaHipRtBase_hpp_HIP)

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(alpaka_queue_cuda_hip_QueueUniformCudaHipRtBase_hpp_CUDA)
#        define alpaka_queue_cuda_hip_QueueUniformCudaHipRtBase_hpp_CUDA
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(alpaka_queue_cuda_hip_QueueUniformCudaHipRtBase_hpp_HIP)
#        define alpaka_queue_cuda_hip_QueueUniformCudaHipRtBase_hpp_HIP
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <memory>

namespace alpaka
{
    namespace ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE
    {
        namespace detail
        {
            //! The CUDA/HIP RT blocking queue implementation.
            class QueueUniformCudaHipRtImpl final
            {
            public:
                ALPAKA_FN_HOST QueueUniformCudaHipRtImpl(DevUniformCudaHipRt const& dev)
                    : m_dev(dev)
                    , m_UniformCudaHipQueue()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.getNativeHandle()));

                    // - [cuda/hip]StreamDefault: Default queue creation flag.
                    // - [cuda/hip]StreamNonBlocking: Specifies that work running in the created queue may run
                    // concurrently with work in queue 0 (the NULL queue),
                    //   and that the created queue should perform no implicit synchronization with queue 0.
                    // Create the queue on the current device.
                    // NOTE: [cuda/hip]StreamNonBlocking is required to match the semantic implemented in the alpaka
                    // CPU queue. It would be too much work to implement implicit default queue synchronization on CPU.

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                        StreamCreateWithFlags)(&m_UniformCudaHipQueue, ALPAKA_API_PREFIX(StreamNonBlocking)));
                }
                QueueUniformCudaHipRtImpl(QueueUniformCudaHipRtImpl&&) = default;
                auto operator=(QueueUniformCudaHipRtImpl&&) -> QueueUniformCudaHipRtImpl& = delete;
                ALPAKA_FN_HOST ~QueueUniformCudaHipRtImpl()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // In case the device is still doing work in the queue when cuda/hip-StreamDestroy() is called, the
                    // function will return immediately and the resources associated with queue will be released
                    // automatically once the device has completed all work in queue.
                    // -> No need to synchronize here.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ALPAKA_API_PREFIX(StreamDestroy)(m_UniformCudaHipQueue));
                }

                [[nodiscard]] auto getNativeHandle() const noexcept
                {
                    return m_UniformCudaHipQueue;
                }

            public:
                DevUniformCudaHipRt const m_dev; //!< The device this queue is bound to.
            private:
                ALPAKA_API_PREFIX(Stream_t) m_UniformCudaHipQueue;
            };

            //! The CUDA RT blocking queue.
            class QueueUniformCudaHipRtBase
                : public concepts::Implements<ConceptCurrentThreadWaitFor, QueueUniformCudaHipRtBase>
                , public concepts::Implements<ConceptQueue, QueueUniformCudaHipRtBase>
                , public concepts::Implements<ConceptGetDev, QueueUniformCudaHipRtBase>
            {
            public:
                ALPAKA_FN_HOST QueueUniformCudaHipRtBase(DevUniformCudaHipRt const& dev)
                    : m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl>(dev))
                {
                }
                ALPAKA_FN_HOST auto operator==(QueueUniformCudaHipRtBase const& rhs) const -> bool
                {
                    return (m_spQueueImpl == rhs.m_spQueueImpl);
                }
                ALPAKA_FN_HOST auto operator!=(QueueUniformCudaHipRtBase const& rhs) const -> bool
                {
                    return !((*this) == rhs);
                }

                [[nodiscard]] auto getNativeHandle() const noexcept
                {
                    return m_spQueueImpl->getNativeHandle();
                }

            public:
                std::shared_ptr<QueueUniformCudaHipRtImpl> m_spQueueImpl;
            };
        } // namespace detail
    } // namespace ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE

    namespace trait
    {
        //! The CUDA/HIP RT non-blocking queue device get trait specialization.
        template<>
        struct GetDev<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::QueueUniformCudaHipRtBase>
        {
            ALPAKA_FN_HOST static auto getDev(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::QueueUniformCudaHipRtBase const& queue)
                -> ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::DevUniformCudaHipRt
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CUDA/HIP RT blocking queue test trait specialization.
        template<>
        struct Empty<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::QueueUniformCudaHipRtBase>
        {
            ALPAKA_FN_HOST static auto empty(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::QueueUniformCudaHipRtBase const& queue) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for queues on non current device.
                ALPAKA_API_PREFIX(Error_t) ret = ALPAKA_API_PREFIX(Success);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = ALPAKA_API_PREFIX(StreamQuery)(queue.getNativeHandle()),
                    ALPAKA_API_PREFIX(ErrorNotReady));
                return (ret == ALPAKA_API_PREFIX(Success));
            }
        };

        //! The CUDA/HIP RT blocking queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<>
        struct CurrentThreadWaitFor<ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::QueueUniformCudaHipRtBase>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(
                ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::QueueUniformCudaHipRtBase const& queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for queues on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamSynchronize)(queue.getNativeHandle()));
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
