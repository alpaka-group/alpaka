/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/predef.h>   // workarounds

// nvcc does not currently support boost correctly.
// boost/utility/detail/result_of_iterate.hpp:148:75: error: invalid use of qualified-name 'std::allocator_traits<_Alloc>::propagate_on_container_swap'
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <queue>        // std::queue
    #include <mutex>        // std::mutex
#else
    #if BOOST_COMP_MSVC
        #pragma warning(push)
        #pragma warning(disable: 4244)  // boost/lockfree/detail/tagged_ptr_ptrcompression.hpp(59): warning C4244: '=': conversion from 'int' to 'boost::lockfree::detail::tagged_ptr<boost::lockfree::detail::freelist_stack<T,Alloc>::freelist_node>::tag_t', possible loss of data
    #endif

    #include <boost/lockfree/queue.hpp>

    #if BOOST_COMP_MSVC
        #pragma warning(pop)
    #endif
#endif

#include <stdexcept>        // std::current_exception
#include <vector>           // std::vector
#include <exception>        // std::runtime_error
#include <utility>          // std::forward
#include <atomic>           // std::atomic
#include <future>           // std::future

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
            //#############################################################################
            //!
            //#############################################################################
            template<
                typename T>
            class ThreadSafeQueue :
                private std::queue<T>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ThreadSafeQueue(
                    std::size_t)
                {}
                //-----------------------------------------------------------------------------
                //! \return If the queue is empty.
                //-----------------------------------------------------------------------------
                auto empty() const
                -> bool
                {
                    return std::queue<T>::empty();
                }
                //-----------------------------------------------------------------------------
                //! Pushes the given value onto the back of the queue.
                //-----------------------------------------------------------------------------
                auto push(
                    T const & t)
                -> void
                {
                    std::lock_guard<std::mutex> lk(m_Mutex);

                    std::queue<T>::push(t);
                }
                //-----------------------------------------------------------------------------
                //! Pops the given value from the front of the queue.
                //-----------------------------------------------------------------------------
                auto pop(
                    T & t)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(m_Mutex);

                    if(std::queue<T>::empty())
                    {
                        return false;
                    }
                    else
                    {
                        t = std::queue<T>::front();
                        std::queue<T>::pop();
                        return true;
                    }
                }

            private:
                std::mutex m_Mutex;
            };
#else
            //#############################################################################
            //!
            //#############################################################################
            template<
                typename T>
            using ThreadSafeQueue = boost::lockfree::queue<T>;
#endif
            //#############################################################################
            //! ITaskPkg.
            //#############################################################################
            // \TODO: Replace with std::packaged_task which was buggy in MSVC 12.
            class ITaskPkg
            {
            public:
                //-----------------------------------------------------------------------------
                //! Runs this task.
                //-----------------------------------------------------------------------------
                auto runTask() noexcept
                -> void
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
                virtual auto run() -> void = 0;

            public:
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual auto setException(
                    std::exception_ptr const & exceptPtr)
                -> void = 0;
            };

            //#############################################################################
            //! TaskPkg with return type.
            //!
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TFnObj The type of the function to execute.
            //! \tparam TFnObjReturn The return type of the TFnObj. Used for class specialization.
            //#############################################################################
            template<
                template<typename TFnObjReturn> class TPromise,
                typename TFnObj,
                typename TFnObjReturn>
            class TaskPkg :
                public ITaskPkg
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                TaskPkg(
                    TFnObj && func) :
                        m_Promise(),
                        m_FnObj(std::forward<TFnObj>(func))
                {}

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual auto run()
                -> void final
                {
                    m_Promise.set_value(this->m_FnObj());
                }
            public:
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual auto setException(
                    std::exception_ptr const & exceptPtr)
                -> void final
                {
                    m_Promise.set_exception(exceptPtr);
                }

                TPromise<TFnObjReturn> m_Promise;
            private:
                TFnObj m_FnObj;
            };

            //#############################################################################
            //! TaskPkg without return type.
            //!
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TFnObj The type of the function to execute.
            //#############################################################################
            template<
                template<typename TFnObjReturn> class TPromise,
                typename TFnObj>
            class TaskPkg<
                TPromise,
                TFnObj,
                void> :
                public ITaskPkg
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                TaskPkg(
                    TFnObj && func) :
                        m_Promise(),
                        m_FnObj(std::forward<TFnObj>(func))
                {}

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual auto run()
                -> void final
                {
                    this->m_FnObj();
                    m_Promise.set_value();
                }
            public:
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual auto setException(
                    std::exception_ptr const & exceptPtr)
                -> void final
                {
                    m_Promise.set_exception(exceptPtr);
                }

                TPromise<void> m_Promise;
            private:
                TFnObj m_FnObj;
            };

            //#############################################################################
            //! ConcurrentExecPool using yield.
            //!
            //! \tparam TConcurrentExec The type of concurrent executor (for example std::thread).
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TYield The type is required to have a static method "void yield()" to yield the current thread if there is no work.
            //! \tparam TMutex Unused. The mutex type used for locking threads.
            //! \tparam TCondVar Unused. The condition variable type used to make the threads wait if there is no work.
            //! \tparam TisYielding Boolean value if the threads should yield instead of wait for a condition variable.
            //#############################################################################
            template<
                typename TSize,
                typename TConcurrentExec,
                template<typename TFnObjReturn> class TPromise,
                typename TYield,
                typename TMutex = void,
                typename TCondVar = void,
                bool TisYielding = true>
            class ConcurrentExecPool final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a concurrent executor pool with a specific number of concurrent executors and a maximum number of queued tasks.
                //!
                //! \param concurrentExecutionCount   The guaranteed number of concurrent executors used in the pool.
                //!                                     This is also the maximum number of tasks worked on concurrently.
                //! \param queueSize  The maximum number of tasks that can be queued for completion.
                //!                     Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(
                    TSize concurrentExecutionCount,
                    TSize queueSize = 128u) :
                    m_vConcurrentExecs(),
                    m_qTasks(queueSize),
                    m_bShutdownFlag(false)
                {
                    m_vConcurrentExecs.reserve(concurrentExecutionCount);

                    // Create all concurrent executors.
                    for(size_t concurrentExec(0u); concurrentExec < concurrentExecutionCount; ++concurrentExec)
                    {
                        m_vConcurrentExecs.emplace_back(std::bind(&ConcurrentExecPool::concurrentExecFn, this));
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool const &) -> ConcurrentExecPool & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool &&) -> ConcurrentExecPool & = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                ~ConcurrentExecPool()
                {
                    // Signal that concurrent executors should not perform any new work
                    m_bShutdownFlag.store(true);

                    joinAllConcurrentExecs();

                    auto currentTaskPackage(std::unique_ptr<ITaskPkg>{nullptr});

                    // Signal to each incomplete task that it will not complete due to pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        auto const except(std::runtime_error("Could not perform task before ConcurrentExecPool destruction"));
                        currentTaskPackage->setException(std::make_exception_ptr(except));
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the pool in First In First Out (FIFO) order.
                //!
                //! \tparam TFnObj   The function type.
                //! \param task     Function object to be called on the pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \tparam TArgs   The argument types pack.
                //! \param args     Arguments for task, cannot be moved.
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //!
                //! \return Signals when the task has completed with either success or an exception.
                //!         Also results in an exception if the pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<
                    typename TFnObj,
                    typename ... TArgs>
                auto enqueueTask(
                    TFnObj && task,
                    TArgs && ... args)
                -> typename std::result_of< decltype(&TPromise<typename std::result_of<TFnObj(TArgs...)>::type>::get_future)(TPromise<typename std::result_of<TFnObj(TArgs...)>::type>) >::type
                {
                    auto boundTask(std::bind(std::forward<TFnObj>(task), std::forward<TArgs>(args)...));

                    // Return type of the function object, can be void via specialization of TaskPkg.
                    using FnObjReturn = typename std::result_of<TFnObj(TArgs...)>::type;
                    using TaskPackage = TaskPkg<TPromise, decltype(boundTask), FnObjReturn>;
                    // Ensures no memory leak if push throws.
                    // \TODO: C++14 std::make_unique would be better.
                    auto packagePtr(std::unique_ptr<TaskPackage>(new TaskPackage(std::move(boundTask))));

                    auto future(packagePtr->m_Promise.get_future());

                    m_qTasks.push(static_cast<ITaskPkg *>(packagePtr.get()));

                    // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                    packagePtr.release();

                    return future;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of concurrent executors available.
                //-----------------------------------------------------------------------------
                auto getConcurrentExecutionCount() const
                -> TSize
                {
                    return m_vConcurrentExecs.size();
                }
                //-----------------------------------------------------------------------------
                //! \return If the work queue is empty.
                //-----------------------------------------------------------------------------
                auto isQueueEmpty() const
                -> bool
                {
#if BOOST_COMP_GNUC
                    return const_cast<ThreadSafeQueue<ITaskPkg *> &>(m_qTasks).empty();
#else
                    return m_qTasks.empty();
#endif
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the concurrent executors are executing.
                //-----------------------------------------------------------------------------
                void concurrentExecFn()
                {
                    // Checks whether pool is being destroyed, if so, stop running.
                    while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                    {
                        auto currentTaskPackage(std::unique_ptr<ITaskPkg>{nullptr});

                        // Use popTask so we only ever have one reference to the ITaskPkg
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
                void joinAllConcurrentExecs()
                {
                    for(auto && concurrentExec : m_vConcurrentExecs)
                    {
                        concurrentExec.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                auto popTask(
                    std::unique_ptr<ITaskPkg> & out)
                -> bool
                {
                    ITaskPkg * tempPtr(nullptr);

                    if(m_qTasks.pop(tempPtr))
                    {
                        out.reset(tempPtr);
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<TConcurrentExec> m_vConcurrentExecs;
                ThreadSafeQueue<ITaskPkg *> m_qTasks;
                std::atomic<bool> m_bShutdownFlag;
            };

            //#############################################################################
            //! ConcurrentExecPool using a condition variable to wait for new work.
            //!
            //! \tparam TConcurrentExec The type of concurrent executor (for example std::thread).
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TYield Unused. The type is required to have a static method "void yield()" to yield the current thread if there is no work.
            //! \tparam TMutex The mutex type used for locking threads.
            //! \tparam TCondVar The condition variable type used to make the threads wait if there is no work.
            //#############################################################################
            template<
                typename TSize,
                typename TConcurrentExec,
                template<typename TFnObjReturn> class TPromise,
                typename TYield,
                typename TMutex,
                typename TCondVar>
            class ConcurrentExecPool<
                TSize,
                TConcurrentExec,
                TPromise,
                TYield,
                TMutex,
                TCondVar,
                false> final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a concurrent executor pool with a specific number of concurrent executors and a maximum number of queued tasks.
                //!
                //! \param concurrentExecutionCount   The guaranteed number of concurrent executors used in the pool.
                //!                                     This is also the maximum number of tasks worked on concurrently.
                //! \param queueSize  The maximum number of tasks that can be queued for completion.
                //!                     Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(
                    TSize concurrentExecutionCount,
                    TSize queueSize = 128u) :
                    m_vConcurrentExecs(),
                    m_qTasks(queueSize),
                    m_mtxWakeup(),
                    m_bShutdownFlag(false),
                    m_cvWakeup()
                {
                    m_vConcurrentExecs.reserve(concurrentExecutionCount);

                    // Create all concurrent executors.
                    for(TSize concurrentExec(0u); concurrentExec < concurrentExecutionCount; ++concurrentExec)
                    {
                        m_vConcurrentExecs.emplace_back(std::bind(&ConcurrentExecPool::concurrentExecFn, this));
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool const &) -> ConcurrentExecPool & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool &&) -> ConcurrentExecPool & = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                ~ConcurrentExecPool()
                {
                    {
                        std::unique_lock<TMutex> lock(m_mtxWakeup);

                        // Signal that concurrent executors should not perform any new work
                        m_bShutdownFlag = true;
                    }

                    m_cvWakeup.notify_all();

                    joinAllConcurrentExecs();

                    auto currentTaskPackage(std::unique_ptr<ITaskPkg>{nullptr});

                    // Signal to each incomplete task that it will not complete due to pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        auto const except(std::runtime_error("Could not perform task before ConcurrentExecPool destruction"));
                        currentTaskPackage->setException(std::make_exception_ptr(except));
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the pool in First In First Out (FIFO) order.
                //!
                //! \tparam TFnObj   The function type.
                //! \param task     Function object to be called on the pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \tparam TArgs   The argument types pack.
                //! \param args     Arguments for task, cannot be moved.
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //!
                //! \return Signals when the task has completed with either success or an exception.
                //!         Also results in an exception if the pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<
                    typename TFnObj,
                    typename ... TArgs>
                auto enqueueTask(
                    TFnObj && task,
                    TArgs && ... args)
                -> typename std::result_of< decltype(&TPromise<typename std::result_of<TFnObj(TArgs...)>::type>::get_future)(TPromise<typename std::result_of<TFnObj(TArgs...)>::type>) >::type
                {
                    auto boundTask(std::bind(std::forward<TFnObj>(task), std::forward<TArgs>(args)...));

                    // Return type of the function object, can be void via specialization of TaskPkg.
                    using FnObjReturn = typename std::result_of<TFnObj(TArgs...)>::type;
                    using TaskPackage = TaskPkg<TPromise, decltype(boundTask), FnObjReturn>;
                    // Ensures no memory leak if push throws.
                    // \TODO: C++14 std::make_unique would be better.
                    auto packagePtr(std::unique_ptr<TaskPackage>(new TaskPackage(std::move(boundTask))));

                    auto future(packagePtr->m_Promise.get_future());

                    m_qTasks.push(static_cast<ITaskPkg *>(packagePtr.get()));

                    // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                    packagePtr.release();

                    m_cvWakeup.notify_one();

                    return future;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of concurrent executors available.
                //-----------------------------------------------------------------------------
                auto getConcurrentExecutionCount() const
                -> TSize
                {
                    return m_vConcurrentExecs.size();
                }
                //-----------------------------------------------------------------------------
                //! \return If the work queue is empty.
                //-----------------------------------------------------------------------------
                auto isQueueEmpty() const
                -> bool
                {
#if BOOST_COMP_GNUC
                    return const_cast<ThreadSafeQueue<ITaskPkg *> &>(m_qTasks).empty();
#else
                    return m_qTasks.empty();
#endif
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the concurrent executors are executing.
                //-----------------------------------------------------------------------------
                void concurrentExecFn()
                {
                    // Checks whether pool is being destroyed, if so, stop running (lazy check without mutex).
                    while(!m_bShutdownFlag)
                    {
                        auto currentTaskPackage(std::unique_ptr<ITaskPkg>{nullptr});

                        // Use popTask so we only ever have one reference to the ITaskPkg
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        else
                        {
                            std::unique_lock<TMutex> lock(m_mtxWakeup);

                            // If the shutdown flag has been set since the last check, return now.
                            if(m_bShutdownFlag)
                            {
                                return;
                            }

                            m_cvWakeup.wait(lock, [this]() { return ((!m_qTasks.empty()) || m_bShutdownFlag); });
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all concurrent executors.
                //-----------------------------------------------------------------------------
                void joinAllConcurrentExecs()
                {
                    for(auto && concurrentExec : m_vConcurrentExecs)
                    {
                        concurrentExec.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                auto popTask(
                    std::unique_ptr<ITaskPkg> & out)
                -> bool
                {
                    ITaskPkg * tempPtr(nullptr);

                    if(m_qTasks.pop(tempPtr))
                    {
                        out.reset(tempPtr);
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<TConcurrentExec> m_vConcurrentExecs;
                ThreadSafeQueue<ITaskPkg *> m_qTasks;

                TMutex m_mtxWakeup;
                std::atomic<bool> m_bShutdownFlag;
                TCondVar m_cvWakeup;
            };
        }
    }
}
