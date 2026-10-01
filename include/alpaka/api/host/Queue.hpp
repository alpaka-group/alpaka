/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/generic.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/Event.hpp"
#include "alpaka/api/host/exec/OmpBlocks.hpp"
#include "alpaka/api/host/exec/Serial.hpp"
#include "alpaka/api/host/exec/TbbBlocks.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/alignedAlloc.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/internal/globalMem.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <mutex>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        public:
            Queue(
                internal::concepts::DeviceHandle auto device,
                uint32_t const idx,
                uint32_t numIdx,
                alpaka::concepts::QueuePolicyList auto const& policies)
                : m_device(std::move(device))
                , m_sharedCallbackThread{std::make_shared<alpaka::core::CallbackThread>(numIdx)}
                , m_idx(idx)
                , m_numaIdx(numIdx)
                , m_isBlocking(internal::isBlocking(policies.getQueueKind()))
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            ~Queue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            Queue(Queue const&) = delete;
            Queue& operator=(Queue const&) = delete;

            Queue(Queue&&) = delete;
            Queue& operator=(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Queue<Queue>);
            }

            Handle<T_Device> m_device;
            std::shared_ptr<core::CallbackThread> m_sharedCallbackThread;
            uint32_t m_idx = 0u;
            uint32_t m_numaIdx = 0u;
            bool m_isBlocking{false};

            /** Number of blocking tasks currently being executed.
             *
             * This variable is only used if m_isBlocking == true.
             *
             * The value is a depth counter because a blocking task can re-enter the queue, e.g. a deferred
             * SharedBuffer deleter submits the free into the queue from within a task which is already running.
             * The queue is only empty (no task is executed) once the depth reached zero again.
             */
            std::atomic<uint32_t> m_blockingTaskExecutionDepth{0u};

            /** Mutex to ensure sequential execution of tasks and operation if the queue is blocking.
             *
             * For non-blocking queue @c m_workerThread is taking care of the execution order.
             * The mutex is recursive because a task may release resources owned by the queue,
             * e.g. a deferred SharedBuffer deleter submits the free into this queue from within a
             * task which already holds the lock.
             */
            std::recursive_mutex m_mutex;

            /** Submit a task to the queue.
             *
             * Centralizes blocking / non-blocking behavior within the method to keep other code as easy as possible.
             * For a blocking queue this method is NOT giving the control back to the caller until the operation is
             * processed.
             * All internal calls should use this method and not enqueue tasks directly in @c m_workerThread
             */
            template<typename T_Fn>
            auto submit(T_Fn&& fn)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                if(m_isBlocking)
                {
                    std::lock_guard<std::recursive_mutex> lk(m_mutex);
                    ++m_blockingTaskExecutionDepth;

                    /* Decrement on every exit path, also if fn() throws, so that the depth always reflects the
                     * number of blocking tasks which are currently executed. */
                    struct DepthGuard
                    {
                        std::atomic<uint32_t>& depth;

                        ~DepthGuard()
                        {
                            --depth;
                        }
                    } depthGuard{m_blockingTaskExecutionDepth};

                    fn();
                    // silent tsan warnings: The promise is fulfilled directly and only a future which is true is
                    // returned, there can not be a data race in between.
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wtsan"
#endif
                    // return a ready future-like placeholder; reuse CallbackThread interface minimally
                    std::promise<void> p;
                    auto f = p.get_future();
                    p.set_value();
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#endif
                    // to keep the uniform interface with the non-blocking case,
                    // return by moving the f since it is move-only
                    return f;
                }
                // enqueue the task into the worker thread, callers can wait/chain later.
                return m_sharedCallbackThread->submit(std::forward<T_Fn>(fn));
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::Queue id=") + std::to_string(m_idx);
            }

            friend struct internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct internal::Enqueue;

            template<alpaka::onHost::concepts::ThreadSpec T_ThreadSpec>
            void enqueue(T_ThreadSpec const& threadSpec, auto const& kernelBundle)
            {
                static_assert(
                    ALPAKA_TYPEOF(threadSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                auto deviceKind = alpaka::getDeviceKind(m_device);

                /* Only set the thread affinity if we use a blocking queue, else the affinity is already set in the
                 * callback thread. The callback thread affinity will be given to all threads created bya task executed
                 * by the callback thread. */
                bool setThreadAffinity = m_isBlocking;
                submit(
                    [kernelBundle, threadSpec, deviceKind, numIdx = m_numaIdx, setThreadAffinity]()
                    {
                        auto moreLayer = Dict{
                            DictEntry(object::launchedWidthFrameSpec, std::false_type{}),
                            DictEntry(object::api, api::host),
                            DictEntry(object::deviceKind, deviceKind),
                            DictEntry(object::exec, threadSpec.getExecutor())};
                        onAcc::Acc acc = makeAcc(threadSpec, numIdx, setThreadAffinity);
                        acc(kernelBundle, moreLayer);
                    });
            }

            template<alpaka::onHost::concepts::FrameSpec T_FrameSpec>
            void enqueue(T_FrameSpec const& frameSpec, auto const& kernelBundle)
            {
                static_assert(
                    ALPAKA_TYPEOF(frameSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                auto adjustedThreadSpec = internal::adjustThreadSpec(*m_device.get(), frameSpec, kernelBundle);
                auto deviceKind = alpaka::getDeviceKind(m_device);

                /* Only set the thread affinity if we use a blocking queue, else the affinity is already set in the
                 * callback thread. The callback thread affinity will be given to all threads created bya task executed
                 * by the callback thread. */
                bool setThreadAffinity = m_isBlocking;
                submit(
                    [kernelBundle, adjustedThreadSpec, deviceKind, numIdx = m_numaIdx, setThreadAffinity]()
                    {
                        auto moreLayer = Dict{
                            DictEntry(object::launchedWidthFrameSpec, std::true_type{}),
                            DictEntry(object::api, api::host),
                            DictEntry(object::deviceKind, deviceKind),
                            DictEntry(object::exec, adjustedThreadSpec.getExecutor())};
                        onAcc::Acc acc = makeAcc(adjustedThreadSpec, numIdx, setThreadAffinity);
                        acc(kernelBundle, moreLayer);
                    });
            }

            /** execute a task in the queue
             *
             * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive as
             * dependency. In this case the destructure of the queue is not called.
             */
            void enqueueHostFn(auto const& task)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                submit(task);
            }

            void enqueueHostFnDeferred(auto const& task)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                if(m_isBlocking)
                {
                    // a blocking queue must acquire this lock to ensure that all pending host tasks have finished
                    std::lock_guard lock(m_mutex);
                    m_sharedCallbackThread->submit(task);
                }
                else
                {
                    m_sharedCallbackThread->submit(task);
                }
            }

            void enqueueNativeFn(auto const& fn)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                submit([queueId = getNativeHandle(), fn]() { fn(queueId); });
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct internal::IsQueueEmpty;

            /** Checks if the queue is empty
             *
             * If m_isBlocking is true, only tasks will be taken into account, events will be ignored they could not
             * influence the usage of isQueueEmpty. if m_isBlocking is false, events will be taken into account because
             * they are handled as normal tasks.
             *
             * @return true if no tasks is executed else false
             */
            bool isQueueEmpty() const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                if(m_isBlocking)
                {
                    // check if the queue is currently executing a blocking task
                    return m_blockingTaskExecutionDepth.load() == 0u;
                }
                else
                {
                    return m_sharedCallbackThread->isEmpty();
                }
            }

            friend struct internal::GetDevice;

            friend struct internal::Wait;
            friend struct internal::WaitFor;
            friend struct internal::Memcpy;
            friend struct internal::MemcpyDeviceGlobal;
            friend struct internal::Memset;
            friend struct alpaka::internal::GetApi;
            friend struct internal::AllocDeferred;
            friend struct internal::IgnoreVisibility;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Device>
        struct Wait::Op<cpu::Queue<T_Device>>
        {
            void operator()(cpu::Queue<T_Device>& queue) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                /* If empty -> Enqueue an empty task as marker and wait for the future
                 * else there is no need to wait
                 */
                if(queue.isQueueEmpty() == false)
                {
                    queue.submit([]() {}).wait();
                }
            }
        };

        template<typename T_Device, typename T_Event>
        struct Enqueue::Event<cpu::Queue<T_Device>, T_Event>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Event& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                // open a scope to avoid logging during we hold the lock for this class
                {
                    // Setting the event state (e.g. the future) and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> eventLock(event.m_mutex);


                    ++event.m_enqueueCount;

                    auto const enqueueCount = event.m_enqueueCount;

                    /* In case the queue is blocking we can not use queue.submit() because we hold the lock already.
                     * The blocking queue executes the lambda directly which will create a deadlock.
                     */
                    if(queue.m_isBlocking)
                    {
                        /* a blocking queue must acquire this lock to ensure that all pending host tasks
                         * have finished
                         */
                        std::lock_guard<std::recursive_mutex> queueLock(queue.m_mutex);
                        // Nothing to do if it has been re-enqueued to a later position in the queue.
                        if(enqueueCount == event.m_enqueueCount)
                        {
                            if(event.m_timingEnabled)
                                event.m_timestamp = std::chrono::steady_clock::now();
                            event.m_lastReadyEnqueueCount = std::max(enqueueCount, event.m_lastReadyEnqueueCount);
                        }
                        // apply a fulfilled future
                        std::promise<void> p;
                        p.set_value();
                        event.m_future = p.get_future();
                    }
                    else
                    {
                        auto sharedEvent = event.getSharedPtr();
                        // Enqueue a task that only resets the events flag if it is completed.
                        event.m_future = queue.submit(
                            [sharedEvent, enqueueCount]() mutable
                            {
                                auto const timestamp = std::chrono::steady_clock::now();
                                std::unique_lock<std::mutex> lk2(sharedEvent->m_mutex);

                                // Nothing to do if it has been re-enqueued to a later position in the queue.
                                if(enqueueCount == sharedEvent->m_enqueueCount)
                                {
                                    if(sharedEvent->m_timingEnabled)
                                        sharedEvent->m_timestamp = timestamp;
                                    sharedEvent->m_lastReadyEnqueueCount
                                        = std::max(enqueueCount, sharedEvent->m_lastReadyEnqueueCount);
                                }
                            });
                    }
                }
            }
        };

        template<typename T_Device, typename T_Event>
        struct WaitFor::Op<cpu::Queue<T_Device>, T_Event>
        {
            void operator()(cpu::Queue<T_Device>& queue, cpu::Event<T_Device>& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                // open a scope to avoid logging during we hold the lock for this class
                {
                    // Setting the event state and enqueuing it has to be atomic.
                    std::unique_lock<std::mutex> eventLock(event.m_mutex);

                    if(!event.isReady())
                    {
                        /* In case the queue is blocking we can not use queue.submit() because we hold the lock
                         * already. The blocking queue executes the lambda directly which will create a deadlock.
                         */
                        if(queue.m_isBlocking)
                        {
                            /* a blocking queue must acquire this lock to ensure that all pending host tasks
                             * have finished
                             */
                            std::lock_guard<std::recursive_mutex> queueLock(queue.m_mutex);
                            std::shared_future sFuture = event.m_future;
                            eventLock.unlock();
                            sFuture.get();
                        }
                        else
                        {
                            auto sharedEvent = event.getSharedPtr();
                            auto oldFuture = event.m_future;

                            // unlock here to avoid keeping the look during the maybe expensive enqueue of the task
                            eventLock.unlock();
                            // Enqueue a task that waits for the given future of the event.
                            queue.submit([sharedEvent, oldFuture]() { oldFuture.get(); });
                        }
                    }
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            /** Perform data copy.
             *
             * To understand the usage of pitches to shift pointers within the implementation see
             * https://alpaka3.readthedocs.io/en/latest/advanced/datastorage.html#pitches
             */
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, T_Source const& source, T_Extents const& extents)
                const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

                // use always 64bit precision to avoid overflows in the pitch calculations
                auto extentMd = pCast<size_t>(extents);
                if(extentMd.product() == size_t{0u})
                    return;

                /* Get all required properties outside the lambda function to not extend the life-time of the data.
                 * The life-time is not extended to have some life-time behaviours with all backends.
                 */
                void* destPtr = toVoidPtr(alpaka::onHost::data(ALPAKA_FORWARD(dest)));
                void const* srcPtr = toVoidPtr(alpaka::onHost::data(source));

                if constexpr(dim == 1u)
                {
                    queue.submit(
                        [numElementsInX = extentMd.x(), destPtr, srcPtr]()
                        { std::memcpy(destPtr, srcPtr, numElementsInX * sizeof(alpaka::GetValueType_t<T_Dest>)); });
                }
                else
                {
                    // memcpy is implemented as row wise copy therefore the last dimension is not required
                    auto destPitchBytesWithoutColumn = pCast<size_t>(onHost::getPitches(dest).eraseBack());
                    auto sourcePitchBytesWithoutColumn = pCast<size_t>(onHost::getPitches(source).eraseBack());

                    queue.submit(
                        [extentMd, destPtr, srcPtr, destPitchBytesWithoutColumn, sourcePitchBytesWithoutColumn]()
                        {
                            alpaka::concepts::Vector<size_t> auto const dstExtentWithoutColumn
                                = pCast<size_t>(extentMd.eraseBack());

                            meta::ndLoopIncIdx(
                                dstExtentWithoutColumn,
                                [&](auto const& idx)
                                {
                                    std::memcpy(
                                        reinterpret_cast<std::uint8_t*>(destPtr)
                                            + (idx * destPitchBytesWithoutColumn).sum(),
                                        reinterpret_cast<std::uint8_t const*>(srcPtr)
                                            + (idx * sourcePitchBytesWithoutColumn).sum(),
                                        static_cast<size_t>(extentMd.back()) * sizeof(alpaka::GetValueType_t<T_Dest>));
                                });
                        });
                }
            }
        };

        // copy to device global memory
        template<typename T_Device, typename T_Source, typename T_Storage, typename T>
        struct internal::MemcpyDeviceGlobal::
            Op<cpu::Queue<T_Device>, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>, T_Source>
        {
            void operator()(
                cpu::Queue<T_Device>& queue,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
                auto&& source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                auto* destPtr = dest.getHandle(api::host).data();
                void const* srcPtr{nullptr};
                if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(source)>)
                    srcPtr = source;
                else
                    srcPtr = toVoidPtr(alpaka::onHost::data(ALPAKA_FORWARD(source)));
                queue.submit([destPtr, srcPtr]() { std::memcpy(destPtr, srcPtr, sizeof(T)); });
            }
        };

        // copy from device global memory
        template<typename T_Device, typename T_Dest, typename T_Storage, typename T>
        struct internal::MemcpyDeviceGlobal::
            Op<cpu::Queue<T_Device>, T_Dest, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>
        {
            void operator()(
                cpu::Queue<T_Device>& queue,
                auto&& dest,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                void* destPtr{nullptr};
                if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(dest)>)
                    destPtr = dest;
                else
                    destPtr = toVoidPtr(alpaka::onHost::data(ALPAKA_FORWARD(dest)));
                auto const* srcPtr = source.getHandle(api::host).data();
                queue.submit([destPtr, srcPtr]() { std::memcpy(destPtr, srcPtr, sizeof(T)); });
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<cpu::Queue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
                const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

                // use always 64bit precision to avoid overflows in the pitch calculations
                auto extentMd = pCast<size_t>(extents);
                if(extentMd.product() == size_t{0u})
                    return;

                void* destPtr = static_cast<void*>(alpaka::onHost::data(dest));

                if constexpr(dim == 1u)
                {
                    queue.submit(
                        [numElementsInX = extentMd.x(), destPtr, byteValue]()
                        { std::memset(destPtr, byteValue, numElementsInX * sizeof(alpaka::GetValueType_t<T_Dest>)); });
                }
                else
                {
                    // memset is implemented as row wise memset therefore the last dimension is not required
                    auto destPitchBytesWithoutColumn = pCast<size_t>(onHost::getPitches(dest).eraseBack());
                    queue.submit(
                        [extentMd, destPtr, destPitchBytesWithoutColumn, byteValue]()
                        {
                            auto const dstExtentWithoutColumn = extentMd.eraseBack();
                            meta::ndLoopIncIdx(
                                dstExtentWithoutColumn,
                                [&](auto const& idx)
                                {
                                    std::memset(
                                        reinterpret_cast<std::uint8_t*>(destPtr)
                                            + (idx * destPitchBytesWithoutColumn).sum(),
                                        byteValue,
                                        extentMd.back() * sizeof(alpaka::GetValueType_t<T_Dest>));
                                });
                        });
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<cpu::Queue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, T_Value elementValue, T_Extents const& extents)
                const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                               && std::same_as<alpaka::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                // avoid that we pass a SharedBuffer and convert non alpaka data views
                alpaka::concepts::IView<T_Value> auto dataView = makeView(dest);

                alpaka::internal::generic::fill(
                    queue,
                    defaultExecutor(getDevice(queue)),
                    dataView.getSubView(extents),
                    elementValue);
            }
        };

        /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
         * within a queue
         */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocDeferred::Op<T_Type, cpu::Queue<T_Device>, T_Extents>
        {
            static consteval uint32_t highestPowerOfTwo(uint32_t value)
            {
                uint32_t result = 1u;
                while((result << 1u) <= value)
                {
                    result <<= 1u;
                }
                return result;
            }

            auto operator()(cpu::Queue<T_Device>& queue, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                auto device = queue.getDevice();
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
                auto queueDependency = queue.getSharedPtr();

                T_Type* ptr = reinterpret_cast<T_Type*>(alpaka::core::alignedAlloc(alignment, memSizeInByte));
                device->pinPointer(ptr, memSizeInByte);

                // queueDependency is captured to keep the device alive until the memory is deleted
                auto deleter = [ptr, queueDep = std::move(queueDependency)]()
                { queueDep.get()->submit([ptr]() { alpaka::core::alignedFree(alignment, ptr); }); };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};

                ALPAKA_LOG_INFO(
                    onHost::logger::memory + onHost::logger::queue,
                    [&]()
                    {
                        std::stringstream ss;
                        ss << sharedBuffer;
                        return ss.str();
                    });
                return sharedBuffer;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal
