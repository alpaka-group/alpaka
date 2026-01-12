/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Comment this out to switch to tbb::task::suspend implementation. It utilizes sleep, instead of properly waiting
// #define ALPAKA_TBB_BARRIER_USE_MUTEX

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include "alpaka/core/Common.hpp"
#    include "alpaka/grid/Traits.hpp"

#    include <oneapi/tbb/concurrent_vector.h>
#    include <oneapi/tbb/task.h>
#    include <oneapi/tbb/task_group.h>

#    include <atomic>
#    include <chrono>
#    include <iostream>
#    include <random>
#    include <thread>

namespace alpaka::core
{
    namespace tbb
    {
        // A reusable barrier for TBB tasks using suspend/resume
        template<typename TIdx>
        class BarrierThread final
        {
        public:
            explicit BarrierThread(TIdx const& threadCount) : m_threadCount(threadCount)
            {
                assertValueUnsigned(threadCount);
                suspended.reserve(static_cast<unsigned long>(threadCount));
            }

            // Called from inside a task to wait until all have arrived
            auto wait() -> void
            {
                oneapi::tbb::task::suspend(
                    [this](oneapi::tbb::task::suspend_point sp) mutable
                    {
                        // push_back returns an iterator to the inserted element
                        auto it = suspended.push_back(std::move(sp));
                        TIdx count = static_cast<TIdx>(std::distance(suspended.begin(), it) + 1);

                        if(count == m_threadCount)
                        {
                            for(auto& sp2 : suspended)
                            {
                                oneapi::tbb::task::resume(std::move(sp2));
                            }
                        }
                    });
            }

        private:
            TIdx const m_threadCount;
            oneapi::tbb::concurrent_vector<oneapi::tbb::task::suspend_point> suspended;
        };
    } // namespace tbb
} // namespace alpaka::core

#endif
