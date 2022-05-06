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
            {
                std::unique_lock<std::mutex> lock_c{c_mutex};
                if(m_busyThreads == m_threads.size())
                {
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
                                        break;

                                    task = std::move(m_tasks.front());
                                    m_tasks.pop();
                                }

                                task();
                                m_busyThreads--;
                            }
                        }));
                }
            }
            m_cond.notify_one();
            m_busyThreads++;

            return f;
        }

    private:
        std::vector<std::thread> m_threads;
        std::condition_variable m_cond;
        std::mutex m_mutex, c_mutex;
        bool m_stop{false};
        std::queue<taskType> m_tasks;
        std::atomic<unsigned int> m_busyThreads{0};
    };
} // namespace alpaka::core
