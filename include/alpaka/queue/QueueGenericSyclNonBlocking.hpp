/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/sycl/Utility.hpp>
#include <alpaka/wait/Traits.hpp>

#include <CL/sycl.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace alpaka
{
    template <typename TDev>
    class EventGenericSycl;

    //#############################################################################
    //! The SYCL non-blocking queue.
    template <typename TDev>
    class QueueGenericSyclNonBlocking final
    {
        friend struct traits::GetDev<QueueGenericSyclNonBlocking<TDev>>;
        friend struct traits::Empty<QueueGenericSyclNonBlocking<TDev>>;
        template <typename TQueue, typename TTask, typename Sfinae> friend struct traits::Enqueue;
        friend struct traits::CurrentThreadWaitFor<QueueGenericSyclNonBlocking<TDev>>;
        friend struct traits::Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST QueueGenericSyclNonBlocking(TDev const& dev)
        : m_dev{dev}
        , m_queue{dev.m_context, // This is important. In SYCL a device can belong to multiple contexts.
                  dev.m_device, {cl::sycl::property::queue::enable_profiling{}, cl::sycl::property::queue::in_order{}}}
        {}
        //-----------------------------------------------------------------------------
        QueueGenericSyclNonBlocking(QueueGenericSyclNonBlocking const &) = default;
        //-----------------------------------------------------------------------------
        QueueGenericSyclNonBlocking(QueueGenericSyclNonBlocking &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueGenericSyclNonBlocking const &) -> QueueGenericSyclNonBlocking & = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueGenericSyclNonBlocking &&) -> QueueGenericSyclNonBlocking & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(QueueGenericSyclNonBlocking const & rhs) const -> bool
        {
            return (m_queue == rhs.m_queue);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(QueueGenericSyclNonBlocking const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~QueueGenericSyclNonBlocking() = default;

    private:
        TDev m_dev; //!< The device this queue is bound to.
        cl::sycl::queue m_queue; //!< The underlying SYCL queue.
        cl::sycl::event m_event{}; //!< The last event in the dependency graph.
        std::vector<cl::sycl::event> m_dependencies = {}; //!< A list of events this queue should wait for.
        std::shared_ptr<std::shared_mutex> mutable mutex_ptr{std::make_shared<std::shared_mutex>()};
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL non-blocking queue device type trait specialization.
        template<typename TDev>
        struct DevType<QueueGenericSyclNonBlocking<TDev>>
        {
            using type = TDev;
        };

        //#############################################################################
        //! The SYCL non-blocking queue device get trait specialization.
        template<typename TDev>
        struct GetDev<QueueGenericSyclNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(QueueGenericSyclNonBlocking<TDev> const& queue)
            {
                std::shared_lock<std::shared_mutex> lock{*queue.mutex_ptr};
                return queue.m_dev;
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue event type trait specialization.
        template<typename TDev>
        struct EventType<QueueGenericSyclNonBlocking<TDev>>
        {
            using type = EventGenericSycl<TDev>;
        };

        //#############################################################################
        //! The SYCL non-blocking queue enqueue trait specialization.
        template<typename TDev, typename TTask>
        struct Enqueue<QueueGenericSyclNonBlocking<TDev>, TTask>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueGenericSyclNonBlocking<TDev>& queue, TTask const& task) -> void
            {
                using namespace cl::sycl;

                if constexpr(detail::is_sycl_enqueueable<TTask>::value) // Device task
                {
                    std::scoped_lock<std::shared_mutex, std::shared_mutex, std::shared_mutex> lock{*queue.mutex_ptr, *queue.m_dev.mutex_ptr, task.pimpl->mutex};

                    // Remove any completed events from the task's dependencies
                    if(!task.pimpl->dependencies.empty())
                        detail::remove_completed(task.pimpl->dependencies);

                    // Remove any completed events from the device's dependencies
                    if(!queue.m_dev.m_dependencies.empty())
                        detail::remove_completed(queue.m_dev.m_dependencies);
                    
                    // Wait for the remaining uncompleted events the device is supposed to wait for
                    if(!queue.m_dev.m_dependencies.empty())
                        task.pimpl->dependencies.insert(end(task.pimpl->dependencies), begin(queue.m_dev.m_dependencies), end(queue.m_dev.m_dependencies));
                    
                    // Wait for any events this queue is supposed to wait for
                    if(!queue.m_dependencies.empty())
                        task.pimpl->dependencies.insert(end(task.pimpl->dependencies), begin(queue.m_dependencies), end(queue.m_dependencies));

                    // Execute the kernel
                    queue.m_event = queue.m_queue.submit(task);

                    // Remove queue dependencies
                    queue.m_dependencies.clear();
                }
                else // Host task
                {
                    std::scoped_lock<std::shared_mutex, std::shared_mutex> lock{*queue.mutex_ptr, *queue.m_dev.mutex_ptr};

                    // Remove any completed events from the device's dependencies
                    if(!queue.m_dev.m_dependencies.empty())
                        detail::remove_completed(queue.m_dev.m_dependencies);
                    
                    // Wait for the remaining uncompleted events the device is supposed to wait for
                    if(!queue.m_dev.m_dependencies.empty())
                        queue.m_dependencies.insert(end(queue.m_dependencies), begin(queue.m_dev.m_dependencies), end(queue.m_dev.m_dependencies));
                    
                    // Execute host task
                    queue.m_event = queue.m_queue.submit([&queue, task](cl::sycl::handler& cgh)
                    {
                        cgh.depends_on(queue.m_dependencies);
                        cgh.codeplay_host_task(task);
                    });

                    // Remove queue dependencies
                    queue.m_dependencies.clear();
                }
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue test trait specialization.
        template<typename TDev>
        struct Empty<QueueGenericSyclNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto empty(QueueGenericSyclNonBlocking<TDev> const& queue) -> bool
            {
                using namespace cl::sycl;

                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                std::shared_lock<std::shared_mutex> lock{*queue.mutex_ptr};

                return queue.m_event.template get_info<info::event::command_execution_status>() == info::event_command_status::complete;
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
        template<typename TDev>
        struct CurrentThreadWaitFor<QueueGenericSyclNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueGenericSyclNonBlocking<TDev> const& queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                std::unique_lock<std::shared_mutex> lock{*queue.mutex_ptr};

                // SYCL objects are reference counted, so we can just copy the queue here
                auto non_const_queue = queue;
                non_const_queue.m_queue.wait_and_throw();
            }
        };
    }
}

#endif
