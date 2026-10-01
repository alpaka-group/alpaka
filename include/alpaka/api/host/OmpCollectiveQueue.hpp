/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file This contains a special queue implementation for one of our users. All requirements to use this queue are in
 * this single file, this provides the possibility that the file is copied into the user project and adjusted if
 * needed.
 *
 * @attention This file is **NOT** included with `#include <alpaka/alpaka.hpp>`
 */

#pragma once

#include "alpaka/api/host/Device.hpp"
#include "alpaka/api/host/Queue.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/tag.hpp"

#include <optional>

namespace alpaka
{
    namespace queueKind
    {
        /** Special parallel OpenMP queue
         *
         * If this queue is used outside of an OpenMP parallel region is behaving like a blocking queue.
         * If it is used within an OpenMP parallel region it has special behaviors.
         *   - All threads within a parallel section execute must call methods collectively.
         *   - operations wait for tasks enqueued outside the parallel region
         *   - enqueueing a kernel, memcopy, memset or fill:
         *         - if kernel: all threads execute the kernel collectively
         *         - there is no barrier before or after the kernel execution
         *         - take care calling two consecutive kernel, memcopy, memset or fill can lead into a data race
         *   - wait operations: all threads will wait and see the same queue state
         *   - enqueueHostFnDeferred: all threads wait before the operation is executed
         *   - isEmpty: all threads wait to see the same state
         *   - allocDeferred: all threads wait and get the same value
         *
         * Operating on some queue handle in parallel from within and outside the OpenMP parallel region results in
         * undefined behavior.
         *
         * The queue can only be used if the dependency OpenMP is available, e.g. by setting the CMake option
         * ALPAKA_DEP_OMP=ON or using the required compiler flags to activate OpenMP.
         */
        struct OmpCollective : category::QueueKind
        {
            static constexpr auto isBlocking()
            {
                return true;
            }

            static std::string getName()
            {
                return "OmpCollective";
            }
        };

        /** @copydoc OmpCollective */
        constexpr auto ompCollective = OmpCollective{};
    } // namespace queueKind
} // namespace alpaka

namespace alpaka::onHost
{
    namespace omp
    {
        /** Special OpenMP invoke.
         *
         * If the caller thread is within a parallel OpenMP scope, the functor will be invoked by one thread via `omp
         * single`. After the call all participating threads will be synced. The function is a collective functions and
         * must be called by all threads in the parallel scope.
         *
         * If this function is called outside of an OpenMP parallel section the functor is invoked by the caller only.
         * Even if OpenMP is not available the function can be called and will simply invoke the functor.
         */
        inline void invokeSingle(auto&& fn) requires(std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
        {
#if ALPAKA_OMP
            if(::omp_in_parallel() != 0)
            {
#    pragma omp single
                {
                    ALPAKA_FORWARD(fn)();
                }
            }
            else
#endif
                ALPAKA_FORWARD(fn)();
        }

        /** Special OpenMP invoke with return value
         *
         * If the caller thread is within a parallel OpenMP scope, the functor will be invoked by one thread via `omp
         * single`. After the call all participating threads will be synced and all threads will return a copy of the
         * same value. The function is a collective functions and must be called by all threads in the parallel scope.
         *
         * If this function is called outside of an OpenMP parallel section the functor is invoked by the caller only.
         * This operation is synchronizing all threads within a parallel OpenMP region.
         *
         * @return A copy of the return value from the invocation. The return value must be assignable.
         */
        inline auto invokeSingle(auto&& fn) requires(!std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
        {
#if ALPAKA_OMP
            if(::omp_in_parallel() != 0)
            {
                using ReturnType = decltype(ALPAKA_FORWARD(fn)());

                // wrap within an optional in cases the return value has no default constructor
                std::optional<ReturnType> returnValue;

#    pragma omp single copyprivate(returnValue)
                {
                    returnValue = ALPAKA_FORWARD(fn)();
                }
                return returnValue.value();
            }
#endif
            return ALPAKA_FORWARD(fn)();
        }
    } // namespace omp

#if ALPAKA_OMP
    namespace cpu
    {
        /** Special OpenMP parallel queue
         *
         * @copydetails alpaka::queueKind::OmpCollective
         */
        template<typename T_Device>
        struct OmpCollectiveQueue : std::enable_shared_from_this<OmpCollectiveQueue<T_Device>>
        {
        public:
            OmpCollectiveQueue(
                internal::concepts::DeviceHandle auto device,
                uint32_t const idx,
                uint32_t numIdx,
                alpaka::concepts::QueuePolicyList auto const policies)
                : m_parentQueue(std::make_shared<Queue<T_Device>>(std::move(device), idx, numIdx, policies))
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            ~OmpCollectiveQueue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            OmpCollectiveQueue(OmpCollectiveQueue const&) = delete;
            OmpCollectiveQueue& operator=(OmpCollectiveQueue const&) = delete;

            OmpCollectiveQueue(OmpCollectiveQueue&&) = delete;
            OmpCollectiveQueue& operator=(OmpCollectiveQueue&&) = delete;

            bool operator==(OmpCollectiveQueue const& other) const
            {
                return m_parentQueue == other.m_parentQueue;
            }

            bool operator!=(OmpCollectiveQueue const& other) const
            {
                return !(*this == other);
            }

            /** Wait for all tasks enqueued before the parallel region. */
            void waitUntilParentQueueIsEmpty()
            {
                auto& pQueue = *m_parentQueue.get();
                // wait for all tasks enqueued before the parallel region
                while(internal::IgnoreVisibility::Op<ParentType>::isQueueBlockingTaskExecuted(pQueue) != 0u)
                    ;
            }

            void waitForNonOmpParallelOps()
            {
                while(numCollectiveTasksExecuted != 0u)
                    ;
            }

            /* Each thread in a parallel OpenMP section must increase the counter when participating in an operation
             * even if it is not actively executing something.
             */
            std::atomic<uint32_t> numCollectiveTasksExecuted = 0;

        private:
            using ParentType = Queue<T_Device>;
            std::shared_ptr<ParentType> m_parentQueue;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::OmpCollectiveQueue id=") + std::to_string(m_parentQueue->m_idx);
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_parentQueue);
            }

            auto getDevice() const
            {
                return internal::getDevice(*m_parentQueue);
            }

            std::shared_ptr<OmpCollectiveQueue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct internal::GetNativeHandle;
            friend struct internal::Enqueue;
            friend struct internal::IsQueueEmpty;
            friend struct internal::GetDevice;
            friend struct internal::Wait;
            friend struct internal::WaitFor;
            friend struct internal::Memcpy;
            friend struct internal::Fill;
            friend struct internal::MemcpyDeviceGlobal;
            friend struct internal::Memset;
            friend struct alpaka::internal::GetApi;
            friend struct internal::AllocDeferred;
            friend struct internal::MakeQueue;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Device>
        struct IgnoreVisibility::Op<cpu::Queue<T_Device>>
        {
            [[nodiscard]] static bool isQueueBlockingTaskExecuted(cpu::Queue<T_Device>& queue)
            {
                return queue.m_blockingTaskExecutionDepth.load() != 0u;
            }
        };

        namespace omp
        {
            /** One thread is invoking the function.
             *
             * For internal usage only. The function is taking care that the queue is correctly bookkeeping internally.
             */
            template<typename T_Device>
            inline void invokeSingleNowait(cpu::OmpCollectiveQueue<T_Device>& queue, auto&& fn)
                requires(std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
            {
                if(::omp_in_parallel() != 0)
                {
                    queue.waitUntilParentQueueIsEmpty();
                    ++queue.numCollectiveTasksExecuted;
#    pragma omp single nowait
                    {
                        ALPAKA_FORWARD(fn)();
                    }
                    --queue.numCollectiveTasksExecuted;
                    return;
                }

                // wait until all threads within the OpenMP parallel region finished there task
                queue.waitForNonOmpParallelOps();
                ALPAKA_FORWARD(fn)();
            }

            /** One thread is invoking the function and the return value is broadcast if it is non-void.
             *
             * For internal usage only. The function is taking care that the queue is correctly bookkeeping internally.
             *
             * @{
             */
            template<typename T_Device>
            inline auto invokeSingleAndWait(cpu::OmpCollectiveQueue<T_Device>& queue, auto&& fn)
                requires(std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
            {
                if(::omp_in_parallel() != 0)
                {
#    pragma omp single
                    {
                        queue.waitUntilParentQueueIsEmpty();
                        ++queue.numCollectiveTasksExecuted;
                        ALPAKA_FORWARD(fn)();
                        --queue.numCollectiveTasksExecuted;
                    }
                    return;
                }

                // wait until all threads within the OpenMP parallel region finished there task
                queue.waitForNonOmpParallelOps();
                ALPAKA_FORWARD(fn)();
            }

            template<typename T_Device>
            inline auto invokeSingleAndWait(cpu::OmpCollectiveQueue<T_Device>& queue, auto&& fn)
                requires(!std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
            {
                if(::omp_in_parallel() != 0)
                {
                    using ReturnType = decltype(ALPAKA_FORWARD(fn)());

                    // wrap within an optional in cases the return value has no default constructor
                    std::optional<ReturnType> returnValue;

#    pragma omp single copyprivate(returnValue)
                    {
                        queue.waitUntilParentQueueIsEmpty();
                        ++queue.numCollectiveTasksExecuted;
                        returnValue = ALPAKA_FORWARD(fn)();
                        --queue.numCollectiveTasksExecuted;
                    }
                    return returnValue.value();
                }

                // wait until all threads within the OpenMP parallel region finished there task
                queue.waitForNonOmpParallelOps();
                return ALPAKA_FORWARD(fn)();
            }

            /** @} */
        } // namespace omp

        template<typename T_Platform, alpaka::concepts::QueuePolicyList T_QueuePolicyList>
        requires(T_QueuePolicyList::getQueueKind() == queueKind::ompCollective)
        struct MakeQueue::Op<cpu::Device<T_Platform>, T_QueuePolicyList>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_QueuePolicyList queuePolicies) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                auto queueHandle = device.getSharedPtr();
                std::lock_guard<std::mutex> lk{device.m_queuesGuard};

                auto newQueue = std::make_shared<cpu::OmpCollectiveQueue<cpu::Device<T_Platform>>>(
                    std::move(queueHandle),
                    device.m_queueWaitFns.size(),
                    device.m_cpuGroupIdx,
                    queuePolicies);

                std::weak_ptr<cpu::OmpCollectiveQueue<cpu::Device<T_Platform>>> weakPtrToQueue = newQueue;
                device.m_queueWaitFns.emplace_back(
                    [weakPtrToQueue]
                    {
                        if(auto queue = weakPtrToQueue.lock())
                            internal::wait(*queue);
                    });

                return newQueue;
            }
        };

        /** OpenMP barrier is used. */
        template<typename T_Device, typename T_Event>
        struct WaitFor::Op<cpu::OmpCollectiveQueue<T_Device>, T_Event>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, cpu::Event<T_Device>& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                internal::omp::invokeSingleAndWait(
                    queue,
                    [&]
                    {
                        internal::WaitFor::Op<cpu::Queue<T_Device>, ALPAKA_TYPEOF(event)>{}(
                            *queue.m_parentQueue,
                            event);
                    });
            }
        };

        /** OpenMP barrier is used. */
        template<typename T_Device>
        struct Wait::Op<cpu::OmpCollectiveQueue<T_Device>>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                internal::omp::invokeSingleAndWait(
                    queue,
                    [&] { internal::Wait::Op<cpu::Queue<T_Device>>{}(*queue.m_parentQueue); });
            }
        };

        /** OpenMP barrier is used to guarantee all threads in the parallel region seeing the same status.*/
        template<typename T_Device>
        struct IsQueueEmpty::Op<cpu::OmpCollectiveQueue<T_Device>>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                return internal::omp::invokeSingleAndWait(
                    [&] { internal::IsQueueEmpty::Op<cpu::Queue<T_Device>>{}(*queue.m_parentQueue); });
            }
        };

        /** Pre OpenMP barrier is used and operation is executed without barrier at the end */
        template<typename T_Device, typename T_Task>
        struct internal::Enqueue::HostTaskDeferred<cpu::OmpCollectiveQueue<T_Device>, T_Task>
        {
            // same as for Enqueue::HostTask, but not waiting for the task to finish
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);

                // ensure that e.g. SharedBuffer::keepAlive() works as expected
                if(::omp_in_parallel() != 0)
                {
#    pragma omp barrier
                }
                internal::omp::invokeSingleNowait(
                    queue,
                    [&]
                    {
                        internal::Enqueue::HostTaskDeferred<cpu::Queue<T_Device>, T_Task>{}(
                            *queue.m_parentQueue,
                            task);
                    });
            }
        };

        /** NO OpenMP barrier used. */
        template<typename T_Device, typename T_Task>
        struct internal::Enqueue::HostTask<cpu::OmpCollectiveQueue<T_Device>, T_Task>
        {
            /** execute a task in the queue
             *
             * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive as
             * dependency. In this case the destructure of the queue is not called.
             */
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                internal::omp::invokeSingleNowait(
                    queue,
                    [&] { internal::Enqueue::HostTask<cpu::Queue<T_Device>, T_Task>{}(*queue.m_parentQueue, task); });
            }
        };

        /** Pre OpenMP barrier is used and operation is executed without barrier at the end */
        template<typename T_Device, typename T_Event>
        struct Enqueue::Event<cpu::OmpCollectiveQueue<T_Device>, T_Event>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Event& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                // ensure that e.g. SharedBuffer::keepAlive() works as expected
                if(::omp_in_parallel() != 0)
                {
#    pragma omp barrier
                }
                internal::omp::invokeSingleNowait(
                    queue,
                    [&] { internal::Enqueue::Event<cpu::Queue<T_Device>, T_Event>{}(*queue.m_parentQueue, event); });
            }
        };

        /** enqueue and execute a kernel
         *
         * If this method is called from within a parallel OpenMP section all threads must participate in the
         * execution of the kernel. The kernel will be executed with the thread spec which is created by adjusting
         * the frame spec to the kernel bundle. threadSpec and kernelBundle must be the same for all threads in the
         * OpenMP section. If this method is called from outside a parallel OpenMP section, the kernel will be
         * enqueued and executed as usual within a blocking queue.
         */
        template<typename T_Device, alpaka::concepts::KernelBundle T_KernelBundle, typename... T_ThreadSpecArgs>
        struct Enqueue::
            Kernel<cpu::OmpCollectiveQueue<T_Device>, onHost::ThreadSpec<T_ThreadSpecArgs...>, T_KernelBundle>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                onHost::ThreadSpec<T_ThreadSpecArgs...> const& threadSpec,
                T_KernelBundle const& kernelBundle) const
            {
                static_assert(
                    ALPAKA_TYPEOF(threadSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);

                if(::omp_in_parallel() != 0)
                {
                    queue.waitUntilParentQueueIsEmpty();
                    ++queue.numCollectiveTasksExecuted;
                    auto deviceKind = alpaka::getDeviceKind(queue.m_parentQueue->m_device);

                    // This queue is not changing the affinity
                    bool setThreadAffinity = false;

                    auto moreLayer = Dict{
                        DictEntry(object::launchedWidthFrameSpec, std::false_type{}),
                        DictEntry(object::api, api::host),
                        DictEntry(object::deviceKind, deviceKind),
                        DictEntry(object::exec, threadSpec.getExecutor())};
                    onAcc::Acc acc = makeAcc(threadSpec, queue.m_parentQueue->m_numaIdx, setThreadAffinity);

                    acc(kernelBundle, moreLayer);
                    --queue.numCollectiveTasksExecuted;
                }
                else
                {
                    // wait until all threads within the OpenMP parallel region finished there task
                    while(queue.numCollectiveTasksExecuted != 0u)
                        ;
                    queue.m_parentQueue->enqueue(threadSpec, kernelBundle);
                }
            }
        };

        /** enqueue and execute a kernel
         *
         * If this method is called from within a parallel OpenMP section all threads must participate in the
         * execution of the kernel. The kernel will be executed with the thread spec which is created by adjusting
         * the frame spec to the kernel bundle. frameSpec and kernelBundle must be the same for all threads in the
         * OpenMP section. If this method is called from outside a parallel OpenMP section, the kernel will be
         * enqueued and executed as usual within a blocking queue.
         */
        template<typename T_Device, alpaka::concepts::KernelBundle T_KernelBundle, typename... T_FrameSpecArgs>
        struct Enqueue::
            Kernel<cpu::OmpCollectiveQueue<T_Device>, onHost::FrameSpec<T_FrameSpecArgs...>, T_KernelBundle>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                onHost::FrameSpec<T_FrameSpecArgs...> const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                static_assert(
                    ALPAKA_TYPEOF(frameSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                if(::omp_in_parallel() != 0)
                {
                    queue.waitUntilParentQueueIsEmpty();
                    ++queue.numCollectiveTasksExecuted;

                    auto adjustedThreadSpec
                        = internal::adjustThreadSpec(*(queue.m_parentQueue->m_device), frameSpec, kernelBundle);
                    auto deviceKind = alpaka::getDeviceKind(queue.m_parentQueue->m_device);

                    // This queue is not changing the affinity
                    bool setThreadAffinity = false;

                    auto moreLayer = Dict{
                        DictEntry(object::launchedWidthFrameSpec, std::true_type{}),
                        DictEntry(object::api, api::host),
                        DictEntry(object::deviceKind, deviceKind),
                        DictEntry(object::exec, adjustedThreadSpec.getExecutor())};
                    onAcc::Acc acc = makeAcc(adjustedThreadSpec, queue.m_parentQueue->m_numaIdx, setThreadAffinity);

                    acc(kernelBundle, moreLayer);
                    --queue.numCollectiveTasksExecuted;
                }
                else
                {
                    // wait until all threads within the OpenMP parallel region finished there task
                    while(queue.numCollectiveTasksExecuted != 0u)
                        ;
                    queue.m_parentQueue->enqueue(frameSpec, kernelBundle);
                }
            }
        };

        /** NO OpenMP barrier used. */
        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                T_Source const& source,
                T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);

                omp::invokeSingleNowait(
                    queue,
                    [&]
                    {
                        Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>{}(
                            *queue.m_parentQueue,
                            dest,
                            source,
                            extents);
                    });
            }
        };

        /** NO OpenMP barrier used. */
        template<typename T_Device, typename T_Source, typename T_Storage, typename T>
        struct MemcpyDeviceGlobal::
            Op<cpu::OmpCollectiveQueue<T_Device>, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>, T_Source>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
                auto&& source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                internal::omp::invokeSingleNowait(
                    queue,
                    [&]
                    {
                        MemcpyDeviceGlobal::Op<
                            cpu::Queue<T_Device>,
                            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>,
                            T_Source>{}(*queue.m_parentQueue, dest, source);
                    });
            }
        };

        /** NO OpenMP barrier used. */
        template<typename T_Device, typename T_Dest, typename T_Storage, typename T>
        struct MemcpyDeviceGlobal::
            Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                internal::omp::invokeSingleNowait(
                    queue,
                    [&]
                    {
                        internal::MemcpyDeviceGlobal::Op<
                            cpu::Queue<T_Device>,
                            T_Dest,
                            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>{}(
                            *queue.m_parentQueue,
                            dest,
                            source);
                    });
            }
        };

        /** NO OpenMP barrier used. */
        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                uint8_t byteValue,
                T_Extents const& extents) const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                internal::omp::invokeSingleNowait(
                    queue,
                    [&]
                    {
                        internal::Memset::Op<cpu::Queue<T_Device>, T_Dest, T_Extents>{}(
                            *queue.m_parentQueue,
                            dest,
                            byteValue,
                            extents);
                    });
            }
        };

        /** NO OpenMP barrier used. */
        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                T_Value elementValue,
                T_Extents const& extents) const
                requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                         && std::same_as<alpaka::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                /* There is no need to wait for queue tasks before the parallel region or if called from outside for
                 * tasks within the parallel region, this will be done during the kernel call in the fill function.
                 */
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                if(::omp_in_parallel() != 0)
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                    auto executors = supportedExecutors(getDevice(queue), exec::allExecutors);
                    // avoid that we pass a SharedBuffer and convert non alpaka data views
                    alpaka::concepts::IView<T_Value> auto dataView = makeView(dest);

                    /* All threads in the OpenMP using the generic fill which is enqueuing a kernel into
                     * cpu::OmpCollectiveQueue. This guarantee that the fill kernel is suing only those threads which
                     * are in the parallel section.
                     */
                    alpaka::internal::generic::fill(
                        queue,
                        std::get<0>(executors),
                        dataView.getSubView(extents),
                        elementValue);
                }
                else
                {
                    internal::Fill::Op<cpu::Queue<T_Device>, T_Dest, T_Value, T_Extents>{}(
                        *queue.m_parentQueue,
                        dest,
                        elementValue,
                        extents);
                }
            }
        };

        /** OpenMP barrier is used. */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocDeferred::Op<T_Type, cpu::OmpCollectiveQueue<T_Device>, T_Extents>
        {
            auto operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);

                return internal::omp::invokeSingleAndWait(
                    queue,
                    [&]
                    {
                        return internal::AllocDeferred::Op<T_Type, cpu::Queue<T_Device>, T_Extents>{}(
                            *queue.m_parentQueue,
                            extents);
                    });
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::OmpCollectiveQueue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_parentQueue->m_device);
        }
    };
#endif

} // namespace alpaka::internal
