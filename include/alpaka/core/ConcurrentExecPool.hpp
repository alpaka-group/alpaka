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
        //! \tparam TFunc The type of the function to execute.
        //! \tparam TFuncReturn The return type of the TFunc. Used for class specialization.
        //#############################################################################
        template<
            template<typename TFuncReturn> class TPromise,
            typename TFunc,
            typename TFuncReturn>
        class TaskPkg :
            public ITaskPkg
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            TaskPkg(
                TFunc && func) :
                    m_Promise(),
                    m_Func(std::forward<TFunc>(func))
            {}

        private:
            //-----------------------------------------------------------------------------
            //! The execution function.
            //-----------------------------------------------------------------------------
            virtual auto run()
            -> void final
            {
                m_Promise.set_value(this->m_Func());
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

            TPromise<TFuncReturn> m_Promise;
        private:
            TFunc m_Func;
        };

        //#############################################################################
        //! TaskPkg without return type.
        //!
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TFunc The type of the function to execute.
        //#############################################################################
        template<
            template<typename TFuncReturn> class TPromise,
            typename TFunc>
        class TaskPkg<
            TPromise,
            TFunc,
            void> :
            public ITaskPkg
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            TaskPkg(
                TFunc && func) :
                    m_Promise(),
                    m_Func(std::forward<TFunc>(func))
            {}

        private:
            //-----------------------------------------------------------------------------
            //! The execution function.
            //-----------------------------------------------------------------------------
            virtual auto run()
            -> void final
            {
                this->m_Func();
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
            TFunc m_Func;
        };

        //#############################################################################
        //! ConcurrentExecPool using yield.
        //!
        //! \tparam TConcurrentExec The type of concurrent executor (for example std::thread).
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TYield The type is required to have a static method "void yield()" to yield the current thread if there is no work.
        //! \tparam TMutex Unused. The mutex type used for locking threads.
        //! \tparam TCondVar Unused. The condition variable type used to make the threads wait if there is no work.
        //! \tparam TbYield Boolean value if the threads should yield instead of wait for a condition variable.
        //#############################################################################
        template<
            typename TConcurrentExec,
            template<typename TFuncReturn> class TPromise,
            typename TYield,
            typename TMutex = void,
            typename TCondVar = void,
            bool TbYield = true>
        class ConcurrentExecPool final
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
            ConcurrentExecPool(
                UInt uiConcurrentExecutionCount,
                UInt uiQueueSize = 128u) :
                m_vConcurrentExecs(),
                m_qTasks(uiQueueSize),
                m_bShutdownFlag(false)
            {
                m_vConcurrentExecs.reserve(uiConcurrentExecutionCount);

                // Create all concurrent executors.
                for(size_t uiConcurrentExec(0u); uiConcurrentExec < uiConcurrentExecutionCount; ++uiConcurrentExec)
                {
                    m_vConcurrentExecs.emplace_back(std::bind(&ConcurrentExecPool::concurrentExecFunc, this));
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
            //! \tparam TFunc   The function type.
            //! \param task     Function or functor to be called on the pool.
            //!                 Takes an arbitrary number of arguments and arbitrary return type.
            //! \tparam TArgs   The argument types pack.
            //! \param args     Arguments for task, cannot be moved.
            //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
            //!
            //! \return Signals when the task has completed with either success or an exception.
            //!         Also results in an exception if the pool is destroyed before execution has begun.
            //-----------------------------------------------------------------------------
            template<
                typename TFunc,
                typename ... TArgs>
            auto enqueueTask(
                TFunc && task,
                TArgs && ... args)
            -> typename std::result_of< decltype(&TPromise<typename std::result_of<TFunc(TArgs...)>::type>::get_future)(TPromise<typename std::result_of<TFunc(TArgs...)>::type>) >::type
            {
                auto boundTask(std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...));

                // Return type of the functor, can be void via specialization of TaskPkg.
                using FuncReturn = typename std::result_of<TFunc(TArgs...)>::type;
                using TaskPackage = TaskPkg<TPromise, decltype(boundTask), FuncReturn>;
                // Ensures no memory leak if push throws.
                // \TODO: C++14 std::make_unique would be better.
                auto packagePtr(std::unique_ptr<TaskPackage>(new TaskPackage(std::move(boundTask))));

                m_qTasks.push(static_cast<ITaskPkg *>(packagePtr.get()));

                auto future(packagePtr->m_Promise.get_future());

                // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                packagePtr.release();

                return future;
            }
            //-----------------------------------------------------------------------------
            //! \return The number of concurrent executors available.
            //-----------------------------------------------------------------------------
            auto getConcurrentExecutionCount() const
            -> UInt
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
            void concurrentExecFunc()
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
            typename TConcurrentExec,
            template<typename TFuncReturn> class TPromise,
            typename TYield,
            typename TMutex,
            typename TCondVar>
        class ConcurrentExecPool<
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
            //! \param uiConcurrentExecutionCount   The guaranteed number of concurrent executors used in the pool.
            //!                                     This is also the maximum number of tasks worked on concurrently.
            //! \param uiQueueSize  The maximum number of tasks that can be queued for completion.
            //!                     Currently running tasks do not belong to the queue anymore.
            //-----------------------------------------------------------------------------
            ConcurrentExecPool(
                UInt uiConcurrentExecutionCount,
                UInt uiQueueSize = 128u) :
                m_vConcurrentExecs(),
                m_qTasks(uiQueueSize),
                m_mtxWakeup(),
                m_bShutdownFlag(false),
                m_cvWakeup()
            {
                m_vConcurrentExecs.reserve(uiConcurrentExecutionCount);

                // Create all concurrent executors.
                for(size_t uiConcurrentExec(0u); uiConcurrentExec < uiConcurrentExecutionCount; ++uiConcurrentExec)
                {
                    m_vConcurrentExecs.emplace_back(std::bind(&ConcurrentExecPool::concurrentExecFunc, this));
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
            //! \tparam TFunc   The function type.
            //! \param task     Function or functor to be called on the pool.
            //!                 Takes an arbitrary number of arguments and arbitrary return type.
            //! \tparam TArgs   The argument types pack.
            //! \param args     Arguments for task, cannot be moved.
            //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
            //!
            //! \return Signals when the task has completed with either success or an exception.
            //!         Also results in an exception if the pool is destroyed before execution has begun.
            //-----------------------------------------------------------------------------
            template<
                typename TFunc,
                typename ... TArgs>
            auto enqueueTask(
                TFunc && task,
                TArgs && ... args)
            -> typename std::result_of< decltype(&TPromise<typename std::result_of<TFunc(TArgs...)>::type>::get_future)(TPromise<typename std::result_of<TFunc(TArgs...)>::type>) >::type
            {
                auto boundTask(std::bind(std::forward<TFunc>(task), std::forward<TArgs>(args)...));

                // Return type of the functor, can be void via specialization of TaskPkg.
                using FuncReturn = typename std::result_of<TFunc(TArgs...)>::type;
                using TaskPackage = TaskPkg<TPromise, decltype(boundTask), FuncReturn>;
                // Ensures no memory leak if push throws.
                // \TODO: C++14 std::make_unique would be better.
                auto packagePtr(std::unique_ptr<TaskPackage>(new TaskPackage(std::move(boundTask))));

                m_qTasks.push(static_cast<ITaskPkg *>(packagePtr.get()));

                auto future(packagePtr->m_Promise.get_future());

                // No longer in danger, can revoke ownership so m_qTasks is not left with dangling reference.
                packagePtr.release();

                m_cvWakeup.notify_one();

                return future;
            }
            //-----------------------------------------------------------------------------
            //! \return The number of concurrent executors available.
            //-----------------------------------------------------------------------------
            auto getConcurrentExecutionCount() const
            -> UInt
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
            void concurrentExecFunc()
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
