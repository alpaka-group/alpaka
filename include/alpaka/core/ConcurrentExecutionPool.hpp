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
        namespace detail
        {
            //#############################################################################
            //! TaskPackage.
            //#############################################################################
            template<typename TCurrentException>
            class TaskPackage
            {
            public:
                using TExceptionPtr = typename std::result_of<decltype(&TCurrentException::current_exception)()>::type;

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
                        setException(TCurrentException::current_exception());
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
                virtual void setException(TExceptionPtr exceptPtr) = 0;
            };

            //#############################################################################
            //! TaskPackageImpl with return type.
            //#############################################################################
            template<typename TCurrentException, template<typename TFuncReturn> class TPromise, typename TFunc, typename TFuncReturn>
            class TaskPackageImpl :
                public TaskPackage<TCurrentException>
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
                virtual void setException(TExceptionPtr exceptPtr) final
                {
                    m_Promise.set_exception(exceptPtr);
                }

                TPromise<TFuncReturn> m_Promise;
            private:
                TFunc m_Func;
            };

            //#############################################################################
            //! TaskPackageImpl without return type.
            //#############################################################################
            template<typename TCurrentException, template<typename TFuncReturn> class TPromise, typename TFunc>
            struct TaskPackageImpl<TCurrentException, TPromise, TFunc, void> :
                public TaskPackage<TCurrentException>
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
                virtual void setException(TExceptionPtr exceptPtr) final
                {
                    m_Promise.set_exception(exceptPtr);
                }

                TPromise<void> m_Promise;
            private:
                TFunc m_Func;
            };

            //#############################################################################
            //! ConcurrentExecutionPool using yield.
            //#############################################################################
            template<typename TConcurrentExecutor, template<typename TFuncReturn> class TPromise, typename TCurrentException, typename TYield, typename TMutex = void, template<typename TMutex> class TUniqueLock = std::atomic, typename TConditionVariable = void, bool TbYield = true>
            class ConcurrentExecutionPool
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a concurrent executor pool with a specific number of concurrent executors and a maximum number of queued tasks.
                //!
                //! \param uiConcurrentExecutionCount   The guaranteed number of concurrent executors used in the pool. 
                //!                                     This is also the maximum number of tasks worked on concurrently.
                //! \param uiQueueSize  The maximum number of tasks that can be queued for completion.
                //!                     Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool(
                    std::size_t uiConcurrentExecutionCount,
                    std::size_t uiQueueSize = 128) :
                    m_vConcurrentExecutors(),
                    m_bShutdownFlag(false),
                    m_qTasks(uiQueueSize)
                {
                    m_vConcurrentExecutors.reserve(uiConcurrentExecutionCount);

                    // Create all concurrent executors.
                    for(size_t uiConcurrentExecutor(0); uiConcurrentExecutor < uiConcurrentExecutionCount; ++uiConcurrentExecutor)
                    {
                        m_vConcurrentExecutors.emplace_back(&ConcurrentExecutionPool::concurrentEcecutorFunc, this);
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool(ConcurrentExecutionPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool(ConcurrentExecutionPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool & operator=(ConcurrentExecutionPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-assignment.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool & operator=(ConcurrentExecutionPool &&) = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                virtual ~ConcurrentExecutionPool()
                {
                    // Signal that concurrent executors should not perform any new work
                    m_bShutdownFlag.store(true);

                    joinAllConcurrentEcecutor();

                    auto currentTaskPackage(std::unique_ptr<TaskPackage<TCurrentException>>{nullptr});

                    // Signal to each incomplete task that it will not complete due to pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        // Boost is missing make_exception_ptr so we have to throw the exception to get a pointer to it.

                        //auto const except(std::runtime_error("Could not perform task before ConcurrentExecutionPool destruction"));
                        //currentTaskPackage->setException(std::make_exception_ptr(except));

                        try
                        {
                            throw std::runtime_error("Could not perform task before ConcurrentExecutionPool destruction");
                        }
                        catch(...)
                        {
                            currentTaskPackage->setException(TCurrentException::current_exception());
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the pool in First In First Out (FIFO) order.
                //! 
                //! \param task     Function or functor to be called on the pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \param args     Arguments for task, cannot be moved. 
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //! 
                //! \return Signals when the task has completed with either success or an exception. 
                //!         Also results in an exception if the pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<typename TFunc, typename ... TArgs>
                auto enqueueTask(TFunc && task, TArgs && ... args)
                    -> typename std::result_of< decltype(&TPromise<typename std::result_of<TFunc(TArgs...)>::type>::get_future)(TPromise<typename std::result_of<TFunc(TArgs...)>::type>) >::type
                {
                    auto boundTask(std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...));

                    //  Return type of the functor, can be void via specialization of TaskPackageImpl.
                    using TFuncReturn = typename std::result_of<TFunc(TArgs...)>::type;
                    using TTaskPackageType = TaskPackageImpl<TCurrentException, TPromise, decltype(boundTask), TFuncReturn>;
                    // Ensures no memory leak if push throws.
                    // TODO: C++14 std::make_unique would be better.
                    auto packagePtr(std::unique_ptr<TTaskPackageType>(new TTaskPackageType(std::move(boundTask))));

                    m_qTasks.push(static_cast<TaskPackage<TCurrentException> *>(packagePtr.get()));

                    auto future(packagePtr->m_Promise.get_future());

                    // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                    packagePtr.release();

                    return future;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of concurrent executors available.
                //-----------------------------------------------------------------------------
                std::size_t getConcurrentExecutionCount() const
                {
                    return m_vConcurrentExecutors.size();
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the concurrent executors are executing.
                //-----------------------------------------------------------------------------
                void concurrentEcecutorFunc()
                {
                    // Checks whether pool is being destroyed, if so, stop running.
                    while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                    {
                        auto currentTaskPackage(std::unique_ptr<TaskPackage<TCurrentException>>{nullptr});

                        // Use popTask so we only ever have one reference to the TaskPackage
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        else
                        {
                            TYield::yield();
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all concurrent executors.
                //-----------------------------------------------------------------------------
                void joinAllConcurrentEcecutor()
                {
                    for(auto && concurrentExecutor : m_vConcurrentExecutors)
                    {
                        concurrentExecutor.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                bool popTask(std::unique_ptr<TaskPackage<TCurrentException>> & out)
                {
                    TaskPackage<TCurrentException> * tempPtr(nullptr);

                    if(m_qTasks.pop(tempPtr))
                    {
                        out.reset(tempPtr);
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<TConcurrentExecutor> m_vConcurrentExecutors;
                std::atomic<bool> m_bShutdownFlag;
                boost::lockfree::queue<TaskPackage<TCurrentException> *> m_qTasks;
            };

            //#############################################################################
            //! ConcurrentExecutionPool using condition_variable to wait for new work.
            //#############################################################################
            template<typename TConcurrentExecutor, template<typename TFuncReturn> class TPromise, typename TCurrentException, typename TYield, typename TMutex, template<typename TMutex> class TUniqueLock, typename TConditionVariable>
            class ConcurrentExecutionPool<TConcurrentExecutor, TPromise, TCurrentException, TYield, TMutex, TUniqueLock, TConditionVariable, false>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a concurrent executor pool with a specific number of concurrent executors and a maximum number of queued tasks.
                //!
                //! \param uiConcurrentExecutionCount   The guaranteed number of concurrent executors used in the pool. 
                //!                                     This is also the maximum number of tasks worked on concurrently.
                //! \param uiQueueSize  The maximum number of tasks that can be queued for completion.
                //!                     Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool(
                    std::size_t uiConcurrentExecutionCount,
                    std::size_t uiQueueSize = 128) :
                    m_vConcurrentExecutors(),
                    m_bShutdownFlag(false),
                    m_qTasks(uiQueueSize),
                    m_cvWakeup(),
                    m_mtxWakeup()
                {
                    m_vConcurrentExecutors.reserve(uiConcurrentExecutionCount);

                    // Create all concurrent executors.
                    for(size_t uiConcurrentExecutor(0); uiConcurrentExecutor < uiConcurrentExecutionCount; ++uiConcurrentExecutor)
                    {
                        m_vConcurrentExecutors.emplace_back(&ConcurrentExecutionPool::concurrentEcecutorFunc, this);
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool(ConcurrentExecutionPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool(ConcurrentExecutionPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool & operator=(ConcurrentExecutionPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move-assignment.
                //-----------------------------------------------------------------------------
                ConcurrentExecutionPool & operator=(ConcurrentExecutionPool &&) = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                virtual ~ConcurrentExecutionPool()
                {
                    // Signal that concurrent executors should not perform any new work
                    m_bShutdownFlag.store(true);

                    m_cvWakeup.notify_all();

                    joinAllConcurrentEcecutor();

                    auto currentTaskPackage(std::unique_ptr<TaskPackage<TCurrentException>>{nullptr});

                    // Signal to each incomplete task that it will not complete due to pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        // Boost is missing make_exception_ptr so we have to throw the exception to get a pointer to it.

                        //auto const except(std::runtime_error("Could not perform task before ConcurrentExecutionPool destruction"));
                        //currentTaskPackage->setException(std::make_exception_ptr(except));

                        try
                        {
                            throw std::runtime_error("Could not perform task before ConcurrentExecutionPool destruction");
                        }
                        catch(...)
                        {
                            currentTaskPackage->setException(TCurrentException::current_exception());
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the pool in First In First Out (FIFO) order.
                //! 
                //! \param task     Function or functor to be called on the pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \param args     Arguments for task, cannot be moved. 
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //! 
                //! \return Signals when the task has completed with either success or an exception. 
                //!         Also results in an exception if the pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<typename TFunc, typename ... TArgs>
                auto enqueueTask(TFunc && task, TArgs && ... args)
                    -> typename std::result_of< decltype(&TPromise<typename std::result_of<TFunc(TArgs...)>::type>::get_future)(TPromise<typename std::result_of<TFunc(TArgs...)>::type>) >::type
                {
                    auto boundTask(std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...));

                    //  Return type of the functor, can be void via specialization of TaskPackageImpl.
                    using TFuncReturn = typename std::result_of<TFunc(TArgs...)>::type;
                    using TTaskPackageType = TaskPackageImpl<TCurrentException, TPromise, decltype(boundTask), TFuncReturn>;
                    // Ensures no memory leak if push throws.
                    // TODO: C++14 std::make_unique would be better.
                    auto packagePtr(std::unique_ptr<TTaskPackageType>(new TTaskPackageType(std::move(boundTask))));

                    m_qTasks.push(static_cast<TaskPackage<TCurrentException> *>(packagePtr.get()));

                    auto future(packagePtr->m_Promise.get_future());

                    // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                    packagePtr.release();
                                    
                    m_cvWakeup.notify_one();

                    return future;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of concurrent executors available.
                //-----------------------------------------------------------------------------
                std::size_t getConcurrentExecutionCount() const
                {
                    return m_vConcurrentExecutors.size();
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the concurrent executors are executing.
                //-----------------------------------------------------------------------------
                void concurrentEcecutorFunc()
                {
                    // Checks whether pool is being destroyed, if so, stop running.
                    while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                    {
                        auto currentTaskPackage(std::unique_ptr<TaskPackage<TCurrentException>>{nullptr});

                        // Use popTask so we only ever have one reference to the TaskPackage
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        else
                        {
                            TUniqueLock<TMutex> lock(m_mtxWakeup);

                            m_cvWakeup.wait(lock, [this]() { return !m_qTasks.empty() || m_bShutdownFlag; });
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all concurrent executors.
                //-----------------------------------------------------------------------------
                void joinAllConcurrentEcecutor()
                {
                    for(auto && concurrentExecutor : m_vConcurrentExecutors)
                    {
                        concurrentExecutor.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                bool popTask(std::unique_ptr<TaskPackage<TCurrentException>> & out)
                {
                    TaskPackage<TCurrentException> * tempPtr(nullptr);

                    if(m_qTasks.pop(tempPtr))
                    {
                        out.reset(tempPtr);
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<TConcurrentExecutor> m_vConcurrentExecutors;
                std::atomic<bool> m_bShutdownFlag;
                boost::lockfree::queue<TaskPackage<TCurrentException> *> m_qTasks;

                TConditionVariable m_cvWakeup;
                TMutex m_mtxWakeup;
            };
        }
}
