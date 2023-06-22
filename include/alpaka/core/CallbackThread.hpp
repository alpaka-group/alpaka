/* Copyright 2022 Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
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
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_stop = true;
                m_cond.notify_one();
            }

            if(m_thread.joinable())
            {
                if(std::this_thread::get_id() == m_thread.get_id())
                {
                    std::cerr << "ERROR in ~CallbackThread: thread joins itself" << std::endl;
                    std::abort();
                }
                m_thread.join();
            }
        }

        // Note: due to different std lib implementations of packaged_task, the lifetime of the passed function either
        // ends when the packaged_task is destroyed (while the returned future is still alive) or when both are
        // destroyed. Therefore, ensure that a submitted task does not extend the lifetime of any object that
        // (transitively) holds the returned future. E.g. don't capture a shared_ptr to an alpaka object that stores
        // the returned future. This is a cyclic dependency and creates a leak.
        template<typename NullaryFunction>
        auto submit(NullaryFunction&& nf) -> std::future<void>
        {
            static_assert(
                std::is_void_v<std::invoke_result_t<NullaryFunction>>,
                "Submitted function must not have any arguments and return void.");
            return submit(Task{std::forward<NullaryFunction>(nf)});
        }

        auto submit(Task task) -> std::future<void>
        {
            auto f = task.get_future();
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                ++m_tasksInProgress;
                m_tasks.emplace(std::move(task));
                if(!m_thread.joinable())
                    startWorkerThread();
            }
            m_cond.notify_one();
            return f;
        }

        [[nodiscard]] auto empty() const
        {
            return m_tasksInProgress == 0;
        }

    private:
        std::thread m_thread;
        std::condition_variable m_cond;
        std::mutex m_mutex;
        bool m_stop{false};
        std::queue<Task> m_tasks;
        std::atomic<int> m_tasksInProgress{0};

        auto startWorkerThread() -> void
        {
            m_thread = std::thread(
                [this]
                {
                    while(true)
                    {
                        {
                            // Do not move the tasks out of the loop else the lifetime could be extended until the
                            // moment where the callback thread is destructed.
                            Task task;
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
                        --m_tasksInProgress;
                    }
                });
        }
    };
} // namespace alpaka::core
