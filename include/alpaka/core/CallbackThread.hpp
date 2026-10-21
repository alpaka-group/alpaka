/* Copyright 2022 Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/hwloc/utility.hpp"
#include "alpaka/core/config.hpp"

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace alpaka::core
{
    /** A thread queue executing tasks asynchronously.
     *
     * This object should be used as a member of objects which are secured by smart pointers to avoid that a task is
     * taking over the ownership of the callback thread and therefore can destroy itself before all tasks are executed.
     */
    class CallbackThread
    {
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
        // A custom class is used because std::function<F> requires F to be copyable, and std::packaged_task provides a
        // std::future which will keep the task alive and we cannot control the moment the future is set.
        //! \todo with C++23 std::move_only_function should be used
        struct Task
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif
        {
            virtual ~Task() = default;
            virtual void run() = 0;
        };

        template<typename T_Function>
        struct FunctionHolder : Task
        {
            T_Function func;

            template<typename T_FunctionFwd>
            explicit FunctionHolder(T_FunctionFwd&& funcFwd) : func{std::forward<T_FunctionFwd>(funcFwd)}
            {
            }

            void run() override
            {
                // if m_func throws, let it propagate
                func();
            }
        };

        using TaskPackage = std::pair<std::unique_ptr<Task>, std::promise<void>>;

        struct State
        {
            std::queue<TaskPackage> tasks;
            std::mutex mutex;
            std::condition_variable cond;
        };

    public:
        CallbackThread(uint32_t cpuGroupIdx) : m_state(std::make_shared<State>()), m_cpuGroupIdx{cpuGroupIdx}
        {
        }

        CallbackThread() : m_state(std::make_shared<State>())
        {
        }

        ~CallbackThread()
        {
            {
                std::unique_lock<std::mutex> lock{m_state->mutex};
                m_thread.request_stop();
                // wakeup the thread in case it is waiting
                m_state->cond.notify_one();
            }

            if(m_thread.joinable())
            {
                if(std::this_thread::get_id() == m_thread.get_id())
                {
                    /* We can not join ourselves.
                     * We can only end here if a task that the callback thread is executing is capturing the object
                     * which is holding the callback thread.
                     */
                    m_thread.detach();
                }
                else
                    m_thread.join();
            }
        }

        //! It is guaranteed that the task is fully destroyed before the future's result is set.
        template<typename T_NullaryFunction>
        auto submit(T_NullaryFunction&& nf) -> std::future<void>
        {
            using DecayedFunction = std::decay_t<T_NullaryFunction>;
            static_assert(
                std::is_void_v<std::invoke_result_t<DecayedFunction>>,
                "Submitted function must not have any arguments and return void.");

            // FunctionHolder stores a copy of the user's task, but may be constructed from an expiring value to avoid
            // the copy. We do NOT store a reference to the users task, which could dangle if the user isn't careful.
            auto tp = std::pair(
                std::make_unique<FunctionHolder<DecayedFunction>>(std::forward<T_NullaryFunction>(nf)),
                std::promise<void>{});
            auto f = tp.second.get_future();
            {
                std::unique_lock<std::mutex> lock{m_state->mutex};
                m_state->tasks.emplace(std::move(tp));
                if(!m_thread.joinable())
                    startWorkerThread();
                m_state->cond.notify_one();
            }

            return f;
        }

        bool isEmpty() const
        {
            std::unique_lock lock{m_state->mutex};
            return m_state->tasks.empty();
        }

    private:
        std::jthread m_thread;
        /** Hold data shared between this call and the thread processing the tasts. */
        std::shared_ptr<State> m_state;
        uint32_t m_cpuGroupIdx = onHost::internal::hwloc::allDomains;

        auto startWorkerThread() -> void
        {
            m_thread = std::jthread(
                [state = m_state, cpuGroupIdx = m_cpuGroupIdx](std::stop_token st)
                {
                    if(cpuGroupIdx != onHost::internal::hwloc::allDomains)
                        onHost::internal::hwloc::setThreadAffinity(cpuGroupIdx);

                    while(true)
                    {
                        std::promise<void> taskPromise;
                        std::exception_ptr eptr;
                        {
                            // Task is destroyed before promise is updated but after the queue state is up to date.
                            std::unique_ptr<Task> task = nullptr;
                            {
                                std::unique_lock<std::mutex> lock{state->mutex};
                                state->cond.wait(
                                    lock,
                                    [&state, &st] { return st.stop_requested() || !state->tasks.empty(); });

                                if(st.stop_requested() && state->tasks.empty())
                                    break;

                                task = std::move(state->tasks.front().first);
                                taskPromise = std::move(state->tasks.front().second);
                            }
                            assert(task);
                            try
                            {
                                task->run();
                            }
                            catch(...)
                            {
                                eptr = std::current_exception();
                            }
                            // Destroy the task and its captures before removing the queue entry. Otherwise an empty
                            // queue could be observed while a capture destructor is still running.
                            task.reset();
                            {
                                std::unique_lock<std::mutex> lock{state->mutex};
                                // Pop the empty queue entry only after the task destruction has completed.
                                state->tasks.pop();
                            }
                        }
                        // In case the executed tasks is the last task in the queue the waiting threads will see the
                        // queue as empty.
                        if(eptr)
                            taskPromise.set_exception(std::move(eptr));
                        else
                            taskPromise.set_value();
                    }
                });
        }
    };
} // namespace alpaka::core
