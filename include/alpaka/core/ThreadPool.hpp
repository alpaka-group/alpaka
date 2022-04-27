/* Copyright 2022 Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace alpaka::core
{
    class ThreadPool
    {
        using taskType = std::packaged_task<void()>;

    public:
        ThreadPool()
        {
            for(unsigned int i = 0u; i < std::thread::hardware_concurrency(); ++i)
                //! All the threads run the lambda function in loop
                m_threads.emplace_back(std::thread(
                    [this]
                    {
                        taskType task;
                        while(true)
                        {
                            {
                                std::unique_lock<std::mutex> lock{m_mutex};
                                m_cond.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

                                if(m_stop && m_tasks.empty())
                                    return;

                                task = std::move(m_tasks.front());
                                m_tasks.pop();
                            }
                            task();
                        }
                    }));
        }
        ~ThreadPool()
        {
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_stop = true;
            }
            m_cond.notify_all();
            for(auto& thread : m_threads)
                thread.join();
        }
        auto submit(taskType&& newTask) -> std::future<void>
        {
            auto f = newTask.get_future();
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_tasks.emplace(std::move(newTask));
            }
            m_cond.notify_one();
            return f;
        }

    private:
        std::vector<std::thread> m_threads;
        std::condition_variable m_cond;
        std::mutex m_mutex;
        bool m_stop = false;
        std::queue<taskType> m_tasks;
    };
} // namespace alpaka::core
