/* Copyright 2022 Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace alpaka::core
{
    class CallbackThread
    {
        using Task = std::packaged_task<void()>;

    public:
        ~CallbackThread()
        {
            m_stop = true;
            m_cond.notify_one();
            if(m_thread.joinable())
                m_thread.join();
        }

        //! Submits a task to the thread. The lifetime of the task may be as long as the lifetime of the returned
        //! future.
        auto submit(Task&& newTask) -> std::future<void>
        {
            auto f = newTask.get_future();
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_tasks.emplace(std::move(newTask));
                if(!m_thread.joinable())
                    startWorkerThread();
            }
            m_cond.notify_one();
            return f;
        }

        //! Submits any callable as task to the thread. The lifetime of the task is ensured to end when the task
        //! completed and is not extended to the lifetime of the returned future.
        template<typename Callable>
        auto submit(Callable&& callable) -> std::future<void>
        {
            // We wrap the callable into an optional, so we can destroy the callable when it was executed
            // the shared state of a std::future obtained from a std::packaged_task may hold the enqueued task and thus
            // extends the lifetime callable inside the task. This caused a problem e.g. with EventGenericThreadsImpl,
            // which holds a future to a task, which captures a shared pointer to the EventGenericThreadsImpl itself,
            // which results in a cyclic reference of shared pointers and the EventGenericThreadsImpl will leak.
            return submit(Task{[c = std::optional<std::decay_t<Callable>>{std::forward<Callable>(callable)}]() mutable
                               {
                                   c.value()();
                                   c = std::nullopt;
                               }});
        }

        auto taskCount() const -> std::size_t
        {
            // TODO(bgruber): not accurate
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_tasks.size();
        }

    private:
        std::thread m_thread;
        std::condition_variable m_cond;
        mutable std::mutex m_mutex;
        std::atomic<bool> m_stop{false};
        std::queue<Task> m_tasks;

        auto startWorkerThread() -> void
        {
            m_thread = std::thread(
                [this]
                {
                    Task task;
                    while(true)
                    {
                        {
                            std::unique_lock<std::mutex> lock{m_mutex};
                            m_cond.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

                            if(m_stop && m_tasks.empty())
                                break;

                            task = std::move(m_tasks.front());
                            m_tasks.pop();
                        }

                        task();
                    }
                });
        }
    };
} // namespace alpaka::core
