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

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_ACC_NO_CUDA

#include <mutex>                    // std::mutex
#include <condition_variable>       // std::condition_variable

namespace alpaka
{
    namespace core
    {
        namespace threads
        {
            //#############################################################################
            //! A barrier.
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
                    TSize const & numThreadsToWaitFor = 0) :
                    m_numThreadsToWaitFor(numThreadsToWaitFor)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThread(
                    BarrierThread const & other) = delete;/* :
                    m_numThreadsToWaitFor(other.m_numThreadsToWaitFor)
                {}*/
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
                    std::unique_lock<std::mutex> lock(m_mtxBarrier);
                    if(--m_numThreadsToWaitFor == 0)
                    {
                        m_cvAllThreadsReachedBarrier.notify_all();
                    }
                    else
                    {
                        m_cvAllThreadsReachedBarrier.wait(lock, [this] { return m_numThreadsToWaitFor == 0; });
                    }
                }

                //-----------------------------------------------------------------------------
                //! \return The number of threads to wait for.
                //! NOTE: The value almost always is invalid the time you get it.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto getNumThreadsToWaitFor() const
                -> TSize
                {
                    return m_numThreadsToWaitFor;
                }

                //-----------------------------------------------------------------------------
                //! Resets the number of threads to wait for to the given number.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto reset(
                    TSize const & numThreadsToWaitFor)
                -> void
                {
                    std::lock_guard<std::mutex> lock(m_mtxBarrier);
                    m_numThreadsToWaitFor = numThreadsToWaitFor;
                }

            private:
                std::mutex m_mtxBarrier;
                std::condition_variable m_cvAllThreadsReachedBarrier;
                TSize m_numThreadsToWaitFor;
            };
        }
    }
}
