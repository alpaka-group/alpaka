/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cstddef>  // std::size_t
#include <vector>   // std::vector
#include <exception>// std::runtime_error
#include <utility>  // std::forward
#include <thread>   // std::thread
#include <atomic>   // std::atomic
#include <future>   // std::promise

#include <boost/predef.h>

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4244)  // boost/lockfree/detail/tagged_ptr_ptrcompression.hpp(59): warning C4244: '=': conversion from 'int' to 'boost::lockfree::detail::tagged_ptr<boost::lockfree::detail::freelist_stack<T,Alloc>::freelist_node>::tag_t', possible loss of data
#endif

#include <boost/lockfree/queue.hpp>

#if BOOST_COMP_MSVC
    #pragma warning( pop )
#endif

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            //#############################################################################
            //! TaskPackage.
            //#############################################################################
            class TaskPackage
            {
            public:
                //-----------------------------------------------------------------------------
                //! Runs this task.
                //-----------------------------------------------------------------------------
                void runTask() noexcept
                {
                    try
                    {
                        run();
                    }
                    catch(...)
                    {
                        setException(std::current_exception());
                    }
                }

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual void run() = 0;
            public:
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual void setException(std::exception_ptr exceptPtr) = 0;
            };

            //#############################################################################
            //! TaskPackageImpl with return type.
            //#############################################################################
            template<typename TReturn, typename TFunc>
            class TaskPackageImpl : 
                public TaskPackage
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                TaskPackageImpl(TFunc && func) :
                    m_Promise(),
                    m_Func(std::forward<TFunc>(func))
                {}

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual void run() final
                {
                    m_Promise.set_value(m_Func());
                }
            public:
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual void setException(std::exception_ptr exceptPtr) final
                {
                    m_Promise.set_exception(exceptPtr);
                }

                std::promise<TReturn> m_Promise;
            private:
                TFunc m_Func;
            };

            //#############################################################################
            //! TaskPackageImpl without return type.
            //#############################################################################
            template<typename TFunc>
            struct TaskPackageImpl<void, TFunc> :
                public TaskPackage
            {
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                TaskPackageImpl(TFunc && func) :
                    m_Promise(),
                    m_Func(std::forward<TFunc>(func))
                {}

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual void run() final
                {
                    m_Func();
                    m_Promise.set_value();
                }
            public:
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual void setException(std::exception_ptr except_ptr) final
                {
                    m_Promise.set_exception(except_ptr);
                }

                std::promise<void> m_Promise;
            private:
                TFunc m_Func;
            };

            //#############################################################################
            //! ThreadPool using yield.
            //#############################################################################
            template<bool TbYield = true>
            class ThreadPool
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a thread pool with a specific number of threads and a maximum number of queued tasks.
                //!
                //! \param uiThreadCount    The guaranteed number of threads used in the thread pool. 
                //!                         This is also the maximum number of tasks worked on concurrently.
                //! \param uiQueueSize  The maximum number of tasks that can be queued for completion.
                //!                     Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ThreadPool(
                    std::size_t uiThreadCount = std::thread::hardware_concurrency(),
                    std::size_t uiQueueSize = 128) :
                    m_vThreads(),
                    m_bShutdownFlag(false),
                    m_qTasks(uiQueueSize)
                {
                    m_vThreads.reserve(uiThreadCount);

                    // Create all threads.
                    for(size_t uiThread(0); uiThread < uiThreadCount; ++uiThread)
                    {
                        m_vThreads.emplace_back(&ThreadPool::threadFunc, this);
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ThreadPool(ThreadPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ThreadPool(ThreadPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ThreadPool & operator=(ThreadPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-assignment.
                //-----------------------------------------------------------------------------
                ThreadPool & operator=(ThreadPool &&) = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                virtual ~ThreadPool()
                {
                    // Signal that threads should not perform any new work
                    m_bShutdownFlag.store(true);

                    joinAllThreads();

                    auto currentTaskPackage(std::unique_ptr<TaskPackage>{nullptr});

                    // Signal to each incomplete task that it will not complete due to thread pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        auto const except(std::runtime_error("Could not perform task before thread pool destruction"));
                        currentTaskPackage->setException(std::make_exception_ptr(except));
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the thread pool threads in First In First Out (FIFO) order.
                //! 
                //! \param task     Function or functor to be called on the thread pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \param args     Arguments for task, cannot be moved. 
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //! 
                //! \return Signals when the task has completed with either success or an exception. 
                //! Also results in an exception if the thread pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<typename TFunc, typename ... TArgs>
                auto enqueueTask(TFunc && task, TArgs && ... args)
                    -> std::future<typename std::result_of<TFunc(TArgs...)>::type>
                {
                    //  Return type of the functor, can be void via specialization of task_package_impl.
                    using TReturn = typename std::result_of<TFunc(TArgs...)>::type;

                    auto boundTask(std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...));

                    // Ensures no memory leak if push throws (it shouldn't but to be safe).
                    using TTaskPackageType = TaskPackageImpl<TReturn, decltype(boundTask)>;
                    // TODO: c++14 std::make_unique would be better.
                    auto packagePtr(std::unique_ptr<TTaskPackageType>(new TTaskPackageType(std::move(boundTask))));

                    m_qTasks.push(static_cast<TaskPackage *>(packagePtr.get()));

                   auto future(packagePtr->m_Promise.get_future());

                    // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                    packagePtr.release();

                    return future;
                };
                //-----------------------------------------------------------------------------
                //! \return The number of threads available.
                //-----------------------------------------------------------------------------
                std::size_t getThreadCount() const
                {
                    return m_vThreads.size();
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the threads are executing.
                //-----------------------------------------------------------------------------
                void threadFunc()
                {
                    // Checks whether parent thread pool is being destroyed, if so, stop running.
                    while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                    {
                        auto currentTaskPackage(std::unique_ptr<TaskPackage>{nullptr});

                        // Use popTask so we only ever have one reference to the TaskPackage
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        else
                        {
                            std::this_thread::yield();
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all threads.
                //-----------------------------------------------------------------------------
                void joinAllThreads()
                {
                    for(auto && thread : m_vThreads)
                    {
                        thread.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                bool popTask(std::unique_ptr<TaskPackage> & out)
                {
                    TaskPackage * tempPtr(nullptr);

                    if(m_qTasks.pop(tempPtr))
                    {
                        out.reset(tempPtr);
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<std::thread> m_vThreads;
                std::atomic<bool> m_bShutdownFlag;
                boost::lockfree::queue<TaskPackage *> m_qTasks;
            };

            //#############################################################################
            //! ThreadPool using condition_variable to wait for new work.
            //#############################################################################
            template<>
            class ThreadPool<false>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a thread pool with a specific number of threads and a maximum number of queued tasks.
                //!
                //! \param uiThreadCount    The guaranteed number of threads used in the thread pool. 
                //!                         This is also the maximum number of tasks worked on concurrently.
                //! \param uiQueueSize  The maximum number of tasks that can be queued for completion.
                //!                     Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ThreadPool(
                    std::size_t uiThreadCount = std::thread::hardware_concurrency(),
                    std::size_t uiQueueSize = 128) :
                    m_vThreads(),
                    m_bShutdownFlag(false),
                    m_qTasks(uiQueueSize),
                    m_cvWakeup(),
                    m_mtxWakeup()
                {
                    m_vThreads.reserve(uiThreadCount);

                    // Create all threads.
                    for(size_t uiThread(0); uiThread < uiThreadCount; ++uiThread)
                    {
                        m_vThreads.emplace_back(&ThreadPool::threadFunc, this);
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ThreadPool(ThreadPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ThreadPool(ThreadPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ThreadPool & operator=(ThreadPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-assignment.
                //-----------------------------------------------------------------------------
                ThreadPool & operator=(ThreadPool &&) = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                virtual ~ThreadPool()
                {
                    // Signal that threads should not perform any new work
                    m_bShutdownFlag.store(true);

                    m_cvWakeup.notify_all();

                    joinAllThreads();

                    auto currentTaskPackage(std::unique_ptr<TaskPackage>{nullptr});

                    // Signal to each incomplete task that it will not complete due to thread pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        auto const except(std::runtime_error("Could not perform task before thread pool destruction"));
                        currentTaskPackage->setException(std::make_exception_ptr(except));
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the thread pool threads in First In First Out (FIFO) order.
                //! 
                //! \param task     Function or functor to be called on the thread pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \param args     Arguments for task, cannot be moved. 
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //! 
                //! \return Signals when the task has completed with either success or an exception. 
                //! Also results in an exception if the thread pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<typename TFunc, typename ... TArgs>
                auto enqueueTask(TFunc && task, TArgs && ... args)
                    -> std::future<typename std::result_of<TFunc(TArgs...)>::type>
                {
                    //  Return type of the functor, can be void via specialization of task_package_impl.
                    using TReturn = typename std::result_of<TFunc(TArgs...)>::type;

                    auto boundTask(std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...));

                    // Ensures no memory leak if push throws (it shouldn't but to be safe).
                    using TTaskPackageType = TaskPackageImpl<TReturn, decltype(boundTask)>;
                    // TODO: c++14 std::make_unique would be better.
                    auto packagePtr(std::unique_ptr<TTaskPackageType>(new TTaskPackageType(std::move(boundTask))));

                    m_qTasks.push(static_cast<TaskPackage *>(packagePtr.get()));

                   auto future(packagePtr->m_Promise.get_future());

                    // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                    packagePtr.release();

                    m_cvWakeup.notify_one();

                    return future;
                };
                //-----------------------------------------------------------------------------
                //! \return The number of threads available.
                //-----------------------------------------------------------------------------
                std::size_t getThreadCount() const
                {
                    return m_vThreads.size();
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the threads are executing.
                //-----------------------------------------------------------------------------
                void threadFunc()
                {
                    // Checks whether parent thread pool is being destroyed, if so, stop running.
                    while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                    {
                        auto currentTaskPackage(std::unique_ptr<TaskPackage>{nullptr});

                        // Use popTask so we only ever have one reference to the TaskPackage
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        else
                        {
                            std::unique_lock<std::mutex> lock(m_mtxWakeup);

                            m_cvWakeup.wait(lock, [this]() { return !m_qTasks.empty() || m_bShutdownFlag; });
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all threads.
                //-----------------------------------------------------------------------------
                void joinAllThreads()
                {
                    for(auto && thread : m_vThreads)
                    {
                        thread.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                bool popTask(std::unique_ptr<TaskPackage> & out)
                {
                    TaskPackage * tempPtr(nullptr);

                    if(m_qTasks.pop(tempPtr))
                    {
                        out.reset(tempPtr);
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<std::thread> m_vThreads;
                std::atomic<bool> m_bShutdownFlag;
                boost::lockfree::queue<TaskPackage *> m_qTasks;

                std::condition_variable m_cvWakeup;
                std::mutex m_mtxWakeup;
            };
        }
    }
}
