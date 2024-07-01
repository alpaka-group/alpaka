/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Comment this out to switch to tbb::task::suspend implementation. It utilizes sleep, instead of properly waiting
#define ALPAKA_TBB_BARRIER_USE_MUTEX

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include "alpaka/core/Common.hpp"
#    include "alpaka/grid/Traits.hpp"

#    ifdef ALPAKA_TBB_BARRIER_USE_MUTEX
#        include <condition_variable>
#        include <mutex>
#    else
#        include <oneapi/tbb/task.h>

#        include <atomic>
#    endif

namespace alpaka::core
{
    namespace tbb
    {
        //! A self-resetting barrier.
        template<typename TIdx>
        class BarrierThread final
        {
        public:
            explicit BarrierThread(TIdx const& threadCount)
                : m_threadCount(threadCount)
                , m_curThreadCount(threadCount)
                , m_generation(0)
            {
            }

            //! Waits for all the other threads to reach the barrier.
            auto wait() -> void
            {
                TIdx const generationWhenEnteredTheWait = m_generation;
#    ifdef ALPAKA_TBB_BARRIER_USE_MUTEX
                std::unique_lock<std::mutex> lock(m_mtxBarrier);
#    endif
                if(--m_curThreadCount == 0)
                {
                    m_curThreadCount = m_threadCount;
                    ++m_generation;
#    ifdef ALPAKA_TBB_BARRIER_USE_MUTEX
                    m_cvAllThreadsReachedBarrier.notify_all();
#    endif
                }
                else
                {
#    ifdef ALPAKA_TBB_BARRIER_USE_MUTEX
                    m_cvAllThreadsReachedBarrier.wait(
                        lock,
                        [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
#    else
                    oneapi::tbb::task::suspend(
                        [&generationWhenEnteredTheWait, this](oneapi::tbb::task::suspend_point tag)
                        {
                            while(generationWhenEnteredTheWait == this->m_generation)
                            {
                                // sleep for 100 microseconds
                                usleep(100);
                            }
                            oneapi::tbb::task::resume(tag);
                        });
#    endif
                }
            }

        private:
#    ifdef ALPAKA_TBB_BARRIER_USE_MUTEX
            std::mutex m_mtxBarrier;
            std::condition_variable m_cvAllThreadsReachedBarrier;
#    endif
            TIdx const m_threadCount;
#    ifdef ALPAKA_TBB_BARRIER_USE_MUTEX
            TIdx m_curThreadCount;
            TIdx m_generation;
#    else
            std::atomic<TIdx> m_curThreadCount;
            std::atomic<TIdx> m_generation;
            oneapi::tbb::task::suspend_point m_tag;
#    endif
        };
    } // namespace tbb
} // namespace alpaka::core

#endif
