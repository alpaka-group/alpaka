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

// Uncomment this to disable the standard spinlock behaviour of the threads
//#define ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_ACC_NO_CUDA

#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
    #include <mutex>                // std::mutex
    #include <condition_variable>   // std::condition_variable
#else
    #include <atomic>               // std::atomic
    #include <thread>               // std::this_thread::yield
#endif

namespace alpaka
{
    namespace core
    {
        namespace threads
        {
            //#############################################################################
            //! A self-resetting barrier.
            //#############################################################################
            template<
                typename TSize>
            class BarrierThread final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA explicit BarrierThread(
                    TSize const & threadCount) :
                    m_threadCount(threadCount),
                    m_curThreadCount(threadCount),
                    m_generation(0)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThread(
                    BarrierThread const & other) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThread(BarrierThread &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierThread const &) -> BarrierThread & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierThread &&) -> BarrierThread & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA ~BarrierThread() = default;

                //-----------------------------------------------------------------------------
                //! Waits for all the other threads to reach the barrier.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto wait()
                -> void
                {
                    TSize const generationWhenEnteredTheWait = m_generation;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                    std::unique_lock<std::mutex> lock(m_mtxBarrier);
#endif
                    if(--m_curThreadCount == 0)
                    {
                        m_curThreadCount = m_threadCount;
                        ++m_generation;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                        m_cvAllThreadsReachedBarrier.notify_all();
#endif
                    }
                    else
                    {
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                        m_cvAllThreadsReachedBarrier.wait(lock, [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
#else
                        while(generationWhenEnteredTheWait == m_generation)
                        {
                            std::this_thread::yield();
                        }
#endif
                    }
                }

            private:
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                std::mutex m_mtxBarrier;
                std::condition_variable m_cvAllThreadsReachedBarrier;
#endif
                const TSize m_threadCount;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                TSize m_curThreadCount;
                TSize m_generation;
#else
                std::atomic<TSize> m_curThreadCount;
                std::atomic<TSize> m_generation;
#endif
            };
        }
    }
}
