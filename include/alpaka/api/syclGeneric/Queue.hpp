/* Copyright 2025 Simeon Ehrig, René Widera, Mehmet Yusufoglu, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/Event.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"
#include "alpaka/onHost/trait.hpp"

#include <algorithm>
#include <future>
#include <shared_mutex>
#include <sstream>
#include <type_traits>

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        /** Dispatch a compile time warp size to the kernel
         *
         * The runtime provided warp size of the device is transformed into a compile time warp size.
         * During the kernel (lambda) call in cgh.parallel_for() the lambda must be annotated with
         * `[[sycl::reqd_sub_group_size(WARP_SIZE)]]`. In cases where the warp size is not supported by device a
         * compiler warning will be shown, therefore a second stage during the call of parallel_for() is required where
         * we check if we know based on macro defines provided by the compiler which subgroup sizes (warp size) are
         * supported for the device ther kernel is currently compiled. In cases, where the macro definition to detect
         * the target device is not in the list (file: core/syclConfig.hpp) we allow all subgroup sizes generated from
         * the runtime dispatcher in this trait. This is also the case if we do not compile ahead of time for a device.
         * @attention If a warning `-Wincorrect-sub-group-size` is shown this means we generated a kernel with an
         * unsupported warp size, triggered by the on host runtime dispatch in this trait.
         *
         * The reason why we do not want to execute the runtime dispatch within the parallel_for, equal to what
         * mainline alpaka is doing, is that any kernel instance should have only one code patch to avoid possible
         * register pressure due to a code path which will maybe never called but is generated in the kernel.
         * This complicated approach gives us the guarantee that the runtime device warp size is used during the kernel
         * generation.
         */
        struct Warpsize
        {
            template<alpaka::concepts::DeviceKind T_DeviceKind>
            struct Dispatch
            {
                auto operator()(T_DeviceKind deviceKind, auto&& fn) const;
            };
        };

        template<>
        struct Warpsize::Dispatch<alpaka::deviceKind::Cpu>
        {
            auto operator()(alpaka::deviceKind::Cpu, auto&& fn, uint32_t warpSize) const
            {
                switch(warpSize)
                {
                case 1u:
                    return fn(std::integral_constant<uint32_t, 1u>{});
                case 2u:
                    return fn(std::integral_constant<uint32_t, 2u>{});
                case 4u:
                    return fn(std::integral_constant<uint32_t, 4u>{});
                case 8u:
                    return fn(std::integral_constant<uint32_t, 8u>{});
                case 16u:
                    return fn(std::integral_constant<uint32_t, 16u>{});
                case 32u:
                    return fn(std::integral_constant<uint32_t, 32u>{});
                default:
                    throw std::runtime_error(
                        std::string("Sycl warp size runtime dispatch, unsupported warpSize: ")
                        + std::to_string(warpSize));
                    return fn(std::integral_constant<uint32_t, 1u>{});
                }
            }
        };

        template<>
        struct Warpsize::Dispatch<alpaka::deviceKind::IntelGpu>
        {
            auto operator()(alpaka::deviceKind::IntelGpu, auto&& fn, uint32_t warpSize) const
            {
                switch(warpSize)
                {
                case 8u:
                    return fn(std::integral_constant<uint32_t, 8u>{});
                case 16u:
                    return fn(std::integral_constant<uint32_t, 16u>{});
                case 32u:
                    return fn(std::integral_constant<uint32_t, 32u>{});
                default:
                    throw std::runtime_error(
                        std::string("Sycl warp size runtime dispatch, unsupported warpSize: ")
                        + std::to_string(warpSize));
                    return fn(std::integral_constant<uint32_t, 32u>{});
                }
            }
        };

        template<>
        struct Warpsize::Dispatch<alpaka::deviceKind::AmdGpu>
        {
            auto operator()(alpaka::deviceKind::AmdGpu, auto&& fn, uint32_t warpSize) const
            {
                switch(warpSize)
                {
                case 32u:
                    return fn(std::integral_constant<uint32_t, 32u>{});
                case 64u:
                    return fn(std::integral_constant<uint32_t, 64u>{});
                default:
                    throw std::runtime_error(
                        std::string("Sycl warp size runtime dispatch, unsupported warpSize: ")
                        + std::to_string(warpSize));
                    return fn(std::integral_constant<uint32_t, 32u>{});
                }
            }
        };

        template<>
        struct Warpsize::Dispatch<alpaka::deviceKind::NvidiaGpu>
        {
            auto operator()(alpaka::deviceKind::NvidiaGpu, auto&& fn, uint32_t warpSize) const
            {
                switch(warpSize)
                {
                case 32u:
                    return fn(std::integral_constant<uint32_t, 32u>{});
                default:
                    throw std::runtime_error(
                        std::string("Sycl warp size runtime dispatch, unsupported warpSize: ")
                        + std::to_string(warpSize));
                    return fn(std::integral_constant<uint32_t, 32u>{});
                }
            }
        };

        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        private:
            friend struct alpaka::internal::GetApi;

            template<alpaka::concepts::Vector TVec>
            static constexpr auto vecToSyclRange(TVec vec)
            {
                constexpr auto dim = std::decay_t<TVec>::dim();
                return [&vec]<auto... I>(std::index_sequence<I...>)
                // TODO: check if this is the correct order
                { return sycl::range<dim>(vec[I]...); }(std::make_index_sequence<dim>{});
            };

            inline constexpr auto dispatchWarpSize(auto&& fn) const
            {
                auto warpSize
                    = internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*m_device.get())>{}(*m_device.get()).warpSize;

                return Warpsize::Dispatch<ALPAKA_TYPEOF(getDeviceKind())>{}(
                    getDeviceKind(),
                    ALPAKA_FORWARD(fn),
                    warpSize);
            }


        public:
            Queue(
                internal::concepts::DeviceHandle auto device,
                uint32_t const idx,
                alpaka::concepts::QueuePolicyList auto const& policies)
                : m_device(std::move(device))
                , m_SharedNativeQueue{std::make_shared<NativeQueue>(
                      onHost::getNativeHandle(m_device).first,
                      onHost::getNativeHandle(m_device).second,
                      policies.getTiming())}
                , m_sharedCallbackThread{std::make_shared<alpaka::core::CallbackThread>()}
                , m_idx(idx)
                , m_isBlocking(internal::isBlocking(policies.getQueueKind()))
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            [[nodiscard]] bool isBlocking() const noexcept
            {
                return m_isBlocking;
            }

            Queue(Queue const&) = delete;
            Queue& operator=(Queue const&) = delete;

            Queue(Queue&&) = delete;
            Queue& operator=(Queue&&) = delete;

            ~Queue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_SharedNativeQueue->getQueue();
            }

            void wait()
            {
                getNativeHandle().wait_and_throw();
            }

            std::string getName() const
            {
                std::stringstream ss;
                ss << "Queue<" << getApi(m_device).getName() << ">";
                ss << " id=" << m_idx;
                return ss.str();
            }

        private:
            friend struct alpaka::internal::GetDeviceType;
            friend struct alpaka::onHost::internal::Enqueue;
            friend struct onHost::internal::AllocDeferred;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            friend struct onHost::internal::GetDevice;

            friend struct alpaka::onHost::internal::WaitFor;

            void waitFor(syclGeneric::Event<T_Device>& event)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                sycl::event sycl_event = event.getNativeHandle();
                sycl::event ev
                    = getNativeHandle().submit([sycl_event](sycl::handler& cgh) { cgh.depends_on(sycl_event); });
                setLastEvent(ev);
            }

            friend struct internal::IsQueueEmpty;

            /** Test of all tasks in the queue are finished
             *
             * @attention We are testing for the last event of last enqueued alpaka event or action. The function
             * cannot check events that were queued directly into the native queue, bypassing alpaka.
             */
            bool isQueueEmpty() const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);

                auto const status = getLastEvent().template get_info<sycl::info::event::command_execution_status>();
                return status == sycl::info::event_command_status::complete;
            }

            //! Thread safe getter for the last sycl event.
            sycl::event getLastEvent() const
            {
                return m_SharedNativeQueue->getLastEvent();
            }

            /** Thread safe setter for the last sycl event
             *
             * To track dependencies this method must be called with any event returned by native sycl calls.
             * The operation is blocking the caller in case the queue is a blocking queue.
             */
            void setLastEvent(sycl::event& ev) const
            {
                m_SharedNativeQueue->setLastEvent(ev);
                if(isBlocking())
                    ev.wait_and_throw();
            }

            void enqueueNativeFn(auto const& fn)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                sycl::event ev = fn(getNativeHandle());
                setLastEvent(ev);
            }

            friend struct alpaka::onHost::internal::Memset;
            friend struct alpaka::onHost::internal::Memcpy;
            friend struct alpaka::onHost::internal::MemcpyDeviceGlobal;
            friend struct alpaka::onHost::internal::Alloc;
            friend struct alpaka::onHost::internal::AllocDeferred;
            friend struct alpaka::onHost::internal::AllocMapped;
            friend struct alpaka::onHost::internal::Fill;

            /** RAII Sycl queue
             *
             * Use this implementation via shared pointer to manage the lifetime independent of the queue and
             * callback thread.
             */
            struct NativeQueue
            {
                NativeQueue(sycl::device device, sycl::context context, timing::Disabled)
                    : m_queue(context, device, {sycl::property::queue::in_order{}})
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                }

                NativeQueue(sycl::device device, sycl::context context, timing::Enabled)
                    : m_queue(
                          context,
                          device,
                          {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}})
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                }

                ~NativeQueue()
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                    try
                    {
                        m_queue.wait_and_throw();
                    }
                    catch(sycl::exception const& err)
                    {
                        std::cerr << "Caught SYCL exception while destructing a SYCL queue: " << err.what() << " ("
                                  << err.code() << ')' << std::endl;
                    }
                    catch(std::exception const& err)
                    {
                        std::cerr << "The following runtime error(s) occurred while destructing a SYCL queue:"
                                  << err.what() << std::endl;
                    }
                }

                sycl::queue getQueue() const
                {
                    return m_queue;
                }

                //! Thread safe getter for the last sycl event.
                sycl::event getLastEvent() const
                {
                    std::shared_lock<std::shared_mutex> lock{m_eventGuard};
                    return m_lastEvent;
                }

                /** Thread safe setter for the last sycl event
                 *
                 * To track dependencies this method must be called with any event returned by native sycl calls.
                 */
                void setLastEvent(sycl::event const& ev) const
                {
                    std::unique_lock<std::shared_mutex> lock{m_eventGuard};
                    m_lastEvent = ev;
                }

                sycl::queue m_queue;

            private:
                /** Event which is representing the last enqueued task/action by alpaka
                 *
                 * @attention You should not use the event directly, use always getLastEvent() or setLastEvent().
                 * Tasks enqueued via the native handle outside of alpaka, will not be tracked by this event, therefore
                 * it can be possible that the queue is not empty but the event is already marked as complete. If you
                 * need to track also tasks enqueued outside of alpaka you should use onHost::wait(auto&&).
                 */
                mutable sycl::event m_lastEvent;
                // secure that two threads can change the event at the same time
                mutable std::shared_mutex m_eventGuard;
            };

            Handle<T_Device> m_device;

            std::shared_ptr<NativeQueue> m_SharedNativeQueue;
            std::shared_ptr<core::CallbackThread> m_sharedCallbackThread;

            uint32_t m_idx = 0u;
            bool m_isBlocking{false};
        };

    } // namespace syclGeneric

    template<typename T_Device, typename T_Task>
    struct internal::Enqueue::HostTask<syclGeneric::Queue<T_Device>, T_Task>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Task const& task) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            sycl::event ev = queue.getNativeHandle().submit(
                [task, sharedCallbackThread = queue.m_sharedCallbackThread](sycl::handler& cgh)
                {
                    cgh.host_task(
                        [sharedCallbackThread = std::move(sharedCallbackThread), task]
                        {
                            auto f = sharedCallbackThread->submit([t = std::move(task)] { t(); });
                            f.wait();
                        });
                });
            queue.setLastEvent(ev);
        }
    };

    template<typename T_Device, typename T_Task>
    struct internal::Enqueue::HostTaskDeferred<syclGeneric::Queue<T_Device>, T_Task>
    {
        // same as for Enqueue::HostTask, but not waiting for the task to finish
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Task const& task) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            sycl::event ev = queue.getNativeHandle().submit(
                [task, sharedCallbackThread = queue.m_sharedCallbackThread](sycl::handler& cgh)
                {
                    cgh.host_task([sharedCallbackThread = std::move(sharedCallbackThread), task]
                                  { sharedCallbackThread->submit([t = std::move(task)] { t(); }); });
                });
            queue.setLastEvent(ev);
        }
    };

    template<typename T_Device, typename T_Event>
    struct internal::Enqueue::Event<syclGeneric::Queue<T_Device>, T_Event>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Event& event) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);

            /* We do not use the last event of the queue itself because creating an emulated event allows to see newly
             * submitted tasks add to the native sycl queue outside alpaka. */
            sycl::event emulatedEvent
                = queue.getNativeHandle().submit([](sycl::handler& cgh) { cgh.single_task([]() {}); });
            // update event
            event.setEvent(emulatedEvent);
            // set last event in the queue
            queue.setLastEvent(emulatedEvent);
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
            const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl::event ev = sycl_queue.memset(
                internal::Data::data(dest),
                byteValue,
                extents.x() * sizeof(alpaka::GetValueType_t<T_Dest>));
            queue.setLastEvent(ev);
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Source const& source,
            T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl::event ev = sycl_queue.memcpy(
                toVoidPtr(internal::Data::data(dest)),
                toVoidPtr(internal::Data::data(source)),
                extents.x() * sizeof(alpaka::GetValueType_t<T_Dest>));
            queue.setLastEvent(ev);
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Fill::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Value, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Value elementValue,
            T_Extents const& extents) const
            requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                     && std::same_as<alpaka::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl::event ev = sycl_queue.fill(internal::Data::data(dest), elementValue, extents.x());
            queue.setLastEvent(ev);
        }
    };

    /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
     * within a queue
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
    struct internal::AllocDeferred::Op<T_Type, syclGeneric::Queue<T_Device>, T_Extents>
    {
        auto operator()(syclGeneric::Queue<T_Device>& queue, T_Extents const& extents) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            auto device = queue.getDevice();
            constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                ALPAKA_TYPEOF(getApi(device)){},
                ALPAKA_TYPEOF(getDeviceKind(device)){});
            auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

            auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
            sycl::queue sycl_queue = queue.getNativeHandle();
            auto queueDependency = queue.getSharedPtr();


            T_Type* ptr = reinterpret_cast<T_Type*>(sycl::aligned_alloc_device(alignment, memSizeInByte, sycl_queue));

            // guarantees that the allocation is blocking the queue if necessary.
            if(queue.isBlocking())
                sycl_queue.wait_and_throw();

            auto deleter = [ptr,
                            sharedCallbackThread = queue.m_sharedCallbackThread,
                            sharedNativeQueue = queue.m_SharedNativeQueue]()
            {
                /* in cases where the deleter lifetime is extended e.g. by using keepAlive() on a buffer it can be that
                 * the queue callback thread is holding the last instance of the deleter. keepAlive() is executed
                 * within a sycl host tasks, it is forbidden to create another host task in a host task, result will be
                 * a deadlock. Therefore, we submit the host task to free the memory first to the callback thread which
                 * is than enqueuing the host task. This means that we can guarantee that the memory is freed after all
                 * work, enqueued at the moment where the deleter is executed, in the sycl queue is finished. The
                 * memory will be freed a little bit later than it could in cases other threads enqueue now kernel,
                 * tasks into the sycl queue while the callback thread is creating the host tasks.
                 */
                sharedCallbackThread->submit(
                    [ptr, sharedNativeQueue]()
                    {
                        sharedNativeQueue->getQueue().submit(
                            [&](sycl::handler& cgh)
                            {
                                cgh.host_task([ptr, syclQueue = sharedNativeQueue->getQueue()]()
                                              { sycl::free(toVoidPtr(ptr), syclQueue); });
                            });
                    });
            };

            auto sharedBuffer = onHost::SharedBuffer{
                deviceDependency,
                ptr,
                extents,
                pitches,
                std::move(deleter),
                Alignment<alignment>{}};
            return sharedBuffer;
        }
    };
} // namespace alpaka::onHost

namespace alpaka::internal

{
    template<typename T_Device>
    struct GetApi::Op<alpaka::onHost::syclGeneric::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

#endif
