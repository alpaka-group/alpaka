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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#include <alpaka/core/Fibers.hpp>

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_ACC_NO_CUDA

#include <mutex>                    // std::unique_lock

namespace alpaka
{
    namespace core
    {
        namespace fibers
        {
            //#############################################################################
            //! A self-resetting barrier.
            // NOTE: We do not use the boost::fibers::barrier because it does not support simple resetting.
            //#############################################################################
            template<
                typename TSize>
            class BarrierFiber
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA explicit BarrierFiber(
                    TSize const & threadCount) :
                    m_threadCount(threadCount),
                    m_curThreadCount(threadCount),
                    m_generation(0)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierFiber(BarrierFiber const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierFiber(BarrierFiber &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierFiber const &) -> BarrierFiber & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierFiber &&) -> BarrierFiber & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BarrierFiber() = default;

                //-----------------------------------------------------------------------------
                //! Waits for all the other fibers to reach the barrier.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto wait()
                -> void
                {
                    const auto generationWhenEnteredTheWait = m_generation;
                    std::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
                    if(--m_threadCount == 0)
                    {
                        m_curThreadCount = m_threadCount;
                        ++m_generation;
                        m_cvAllFibersReachedBarrier.notify_all();
                    }
                    else
                    {
                        m_cvAllFibersReachedBarrier.wait(lock, [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
                    }
                }

            private:
                boost::fibers::mutex m_mtxBarrier;
                boost::fibers::condition_variable m_cvAllFibersReachedBarrier;
                const TSize m_threadCount;
                TSize m_curThreadCount;
                TSize m_generation;
            };
        }
    }
}

#endif
