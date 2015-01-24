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

#include <cstddef>  // std::size_t
#include <vector>   // std::vector
#include <exception>// std::runtime_error
#include <utility>  // std::forward
#include <atomic>   // std::atomic
#include <future>   // std::future

#include <boost/predef.h>

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4244)  // boost/lockfree/detail/tagged_ptr_ptrcompression.hpp(59): warning C4244: '=': conversion from 'int' to 'boost::lockfree::detail::tagged_ptr<boost::lockfree::detail::freelist_stack<T,Alloc>::freelist_node>::tag_t', possible loss of data
#endif

#include <boost/lockfree/queue.hpp>

#if BOOST_COMP_MSVC
    #pragma warning(pop)
#endif

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! ITaskPkg.
        //!
        //! \tparam TCurrentException Must have a static method "current_exception()" that returns the current exception.
        //#############################################################################
        template<
            typename TCurrentException>
        class ITaskPkg
        {
        public:
            using ExceptionPtr = typename std::result_of<decltype(&TCurrentException::current_exception)()>::type;

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
                    setException(
                        TCurrentException::current_exception());
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
            virtual void setException(
                ExceptionPtr exceptPtr) = 0;
        };

        //#############################################################################
        //! TaskPkg with return type.
        //!
        //! \tparam TCurrentException Must have a static method "current_exception()" that returns the current exception.
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TFunc The type of the function to execute.
        //! \tparam TFuncReturn The return type of the TFunc. Used for class specialization.
        //#############################################################################
        template<
            typename TCurrentException, 
            template<typename TFuncReturn> class TPromise, 
            typename TFunc, 
            typename TFuncReturn>
        class TaskPkg :
            public ITaskPkg<TCurrentException>
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
            virtual void run() final
            {
                m_Promise.set_value(m_Func());
            }
        public:
            //-----------------------------------------------------------------------------
            //! Sets an exception.
            //-----------------------------------------------------------------------------
            virtual void setException(
                typename ITaskPkg<TCurrentException>::ExceptionPtr exceptPtr) final
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
        //! \tparam TCurrentException Must have a static method "current_exception()" that returns the current exception.
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TFunc The type of the function to execute.
        //#############################################################################
        template<
            typename TCurrentException, 
            template<typename TFuncReturn> class TPromise, 
            typename TFunc>
        class TaskPkg<
            TCurrentException, 
            TPromise, 
            TFunc, 
            void> :
            public ITaskPkg<TCurrentException>
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
            virtual void run() final
            {
                m_Func();
                m_Promise.set_value();
            }
        public:
            //-----------------------------------------------------------------------------
            //! Sets an exception.
            //-----------------------------------------------------------------------------
            virtual void setException(typename ITaskPkg<TCurrentException>::ExceptionPtr exceptPtr) final
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
        //! \tparam TConcurrentExecutor The type of concurrent executor (for example std::thread). 
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TCurrentException Must have a static method "current_exception()" that returns the current exception.
        //! \tparam TYield The type is required to have a static method "void yield()" to yield the current thread if there is no work.
        //! \tparam TMutex Unused. The mutex type used for locking threads.
        //! \tparam TUniqueLock Unused. The lock type used to lock the TMutex.
        //! \tparam TCondVar Unused. The condition variable type used to make the threads wait if there is no work. Uses the TUniqueLock.
        //! \tparam TbYield Booleam value the threads should yield instead of wait for a condition variable.
        //#############################################################################
        template<
            typename TConcurrentExecutor, 
            template<typename TFuncReturn> class TPromise, 
            typename TCurrentException, 
            typename TYield, 
            typename TMutex = void, 
            template<typename TMutex2> class TUniqueLock = std::atomic, 
            typename TCondVar = void, 
            bool TbYield = true>
        class ConcurrentExecPool
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
                    m_vConcurrentExecutors.emplace_back(&ConcurrentExecPool::concurrentExecutorFunc, this);
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
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            ConcurrentExecPool & operator=(ConcurrentExecPool const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment.
            //-----------------------------------------------------------------------------
            ConcurrentExecPool & operator=(ConcurrentExecPool &&) = delete;

            //-----------------------------------------------------------------------------
            //! Destructor
            //!
            //! Completes any currently running task as normal.
            //! Signals a std::runtime_error exception to any other tasks that were not able to run.
            //-----------------------------------------------------------------------------
            virtual ~ConcurrentExecPool()
            {
                // Signal that concurrent executors should not perform any new work
                m_bShutdownFlag.store(true);

                joinAllConcurrentExecutors();

                auto currentTaskPackage(std::unique_ptr<ITaskPkg<TCurrentException>>{nullptr});

                // Signal to each incomplete task that it will not complete due to pool destruction.
                while(popTask(currentTaskPackage))
                {
                    // Boost is missing make_exception_ptr so we have to throw the exception to get a pointer to it.

                    //auto const except(std::runtime_error("Could not perform task before ConcurrentExecPool destruction"));
                    //currentTaskPackage->setException(std::make_exception_ptr(except));

                    try
                    {
                        throw std::runtime_error("Could not perform task before ConcurrentExecPool destruction");
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

                //  Return type of the functor, can be void via specialization of TaskPkg.
                using FuncReturn = typename std::result_of<TFunc(TArgs...)>::type;
                using TaskPackage = TaskPkg<TCurrentException, TPromise, decltype(boundTask), FuncReturn>;
                // Ensures no memory leak if push throws.
                // \TODO: C++14 std::make_unique would be better.
                auto packagePtr(std::unique_ptr<TaskPackage>(new TaskPackage(std::move(boundTask))));

                m_qTasks.push(static_cast<ITaskPkg<TCurrentException> *>(packagePtr.get()));

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
            void concurrentExecutorFunc()
            {
                // Checks whether pool is being destroyed, if so, stop running.
                while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                {
                    auto currentTaskPackage(std::unique_ptr<ITaskPkg<TCurrentException>>{nullptr});

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
            void joinAllConcurrentExecutors()
            {
                for(auto && concurrentExecutor : m_vConcurrentExecutors)
                {
                    concurrentExecutor.join();
                }
            }
            //-----------------------------------------------------------------------------
            //! Pops a task from the queue.
            //-----------------------------------------------------------------------------
            bool popTask(
                std::unique_ptr<ITaskPkg<TCurrentException>> & out)
            {
                ITaskPkg<TCurrentException> * tempPtr(nullptr);

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
            boost::lockfree::queue<ITaskPkg<TCurrentException> *> m_qTasks;
        };

        //#############################################################################
        //! ConcurrentExecPool using a condition variable to wait for new work.
        //!
        //! \tparam TConcurrentExecutor The type of concurrent executor (for example std::thread). 
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TCurrentException Must have a static method "current_exception()" that returns the current exception.
        //! \tparam TYield Unused. The type is required to have a static method "void yield()" to yield the current thread if there is no work.
        //! \tparam TMutex The mutex type used for locking threads.
        //! \tparam TUniqueLock The lock type used to lock the TMutex.
        //! \tparam TCondVar The condition variable type used to make the threads wait if there is no work. Uses the TUniqueLock.
        //#############################################################################
        template<
            typename TConcurrentExecutor, 
            template<typename TFuncReturn> class TPromise, 
            typename TCurrentException, 
            typename TYield, 
            typename TMutex, 
            template<typename TMutex2> class TUniqueLock, 
            typename TCondVar>
        class ConcurrentExecPool<
            TConcurrentExecutor, 
            TPromise, 
            TCurrentException, 
            TYield, 
            TMutex, 
            TUniqueLock, 
            TCondVar, 
            false>
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
                    m_vConcurrentExecutors.emplace_back(&ConcurrentExecPool::concurrentExecutorFunc, this);
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
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            ConcurrentExecPool & operator=(ConcurrentExecPool const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment.
            //-----------------------------------------------------------------------------
            ConcurrentExecPool & operator=(ConcurrentExecPool &&) = delete;

            //-----------------------------------------------------------------------------
            //! Destructor
            //!
            //! Completes any currently running task as normal.
            //! Signals a std::runtime_error exception to any other tasks that were not able to run.
            //-----------------------------------------------------------------------------
            virtual ~ConcurrentExecPool()
            {
                // Signal that concurrent executors should not perform any new work
                m_bShutdownFlag.store(true);

                m_cvWakeup.notify_all();

                joinAllConcurrentExecutors();

                auto currentTaskPackage(std::unique_ptr<ITaskPkg<TCurrentException>>{nullptr});

                // Signal to each incomplete task that it will not complete due to pool destruction.
                while(popTask(currentTaskPackage))
                {
                    // Boost is missing make_exception_ptr so we have to throw the exception to get a pointer to it.

                    //auto const except(std::runtime_error("Could not perform task before ConcurrentExecPool destruction"));
                    //currentTaskPackage->setException(std::make_exception_ptr(except));

                    try
                    {
                        throw std::runtime_error("Could not perform task before ConcurrentExecPool destruction");
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

                //  Return type of the functor, can be void via specialization of TaskPkg.
                using FuncReturn = typename std::result_of<TFunc(TArgs...)>::type;
                using TaskPackage = TaskPkg<TCurrentException, TPromise, decltype(boundTask), FuncReturn>;
                // Ensures no memory leak if push throws.
                // \TODO: C++14 std::make_unique would be better.
                auto packagePtr(std::unique_ptr<TaskPackage>(new TaskPackage(std::move(boundTask))));

                m_qTasks.push(static_cast<ITaskPkg<TCurrentException> *>(packagePtr.get()));

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
            void concurrentExecutorFunc()
            {
                // Checks whether pool is being destroyed, if so, stop running.
                while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                {
                    auto currentTaskPackage(std::unique_ptr<ITaskPkg<TCurrentException>>{nullptr});

                    // Use popTask so we only ever have one reference to the ITaskPkg
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
            void joinAllConcurrentExecutors()
            {
                for(auto && concurrentExecutor : m_vConcurrentExecutors)
                {
                    concurrentExecutor.join();
                }
            }
            //-----------------------------------------------------------------------------
            //! Pops a task from the queue.
            //-----------------------------------------------------------------------------
            bool popTask(
                std::unique_ptr<ITaskPkg<TCurrentException>> & out)
            {
                ITaskPkg<TCurrentException> * tempPtr(nullptr);

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
            boost::lockfree::queue<ITaskPkg<TCurrentException> *> m_qTasks;

            TCondVar m_cvWakeup;
            TMutex m_mtxWakeup;
        };
    }
}
