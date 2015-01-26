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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_ACC_NO_CUDA

#include <mutex>                    // std::mutex
#include <condition_variable>       // std::condition_variable

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            //#############################################################################
            //! A barrier.
            //#############################################################################
            class ThreadBarrier
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA explicit ThreadBarrier(
                    std::size_t const uiNumThreadsToWaitFor = 0) :
                    m_uiNumThreadsToWaitFor(uiNumThreadsToWaitFor)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA ThreadBarrier(
                    ThreadBarrier const & other) :
                    m_uiNumThreadsToWaitFor(other.m_uiNumThreadsToWaitFor)
                {}
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA ThreadBarrier(ThreadBarrier &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA ThreadBarrier & operator=(ThreadBarrier const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL    // threads/Barrier.hpp(66): error : the declared exception specification is incompatible with the generated one
                ALPAKA_FCT_ACC_NO_CUDA virtual ~ThreadBarrier() = default;
#else
                ALPAKA_FCT_ACC_NO_CUDA virtual ~ThreadBarrier() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Waits for all the other threads to reach the barrier.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void wait()
                {
                    std::unique_lock<std::mutex> lock(m_mtxBarrier);
                    if(--m_uiNumThreadsToWaitFor == 0)
                    {
                        m_cvAllThreadsReachedBarrier.notify_all();
                    }
                    else
                    {
                        m_cvAllThreadsReachedBarrier.wait(lock, [this] { return m_uiNumThreadsToWaitFor == 0; });
                    }
                }

                //-----------------------------------------------------------------------------
                //! \return The number of threads to wait for.
                //! NOTE: The value almost always is invalid the time you get it.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA std::size_t getNumThreadsToWaitFor() const
                {
                    return m_uiNumThreadsToWaitFor;
                }

                //-----------------------------------------------------------------------------
                //! Resets the number of threads to wait for to the given number.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void reset(
                    std::size_t const uiNumThreadsToWaitFor)
                {
                    std::lock_guard<std::mutex> lock(m_mtxBarrier);
                    m_uiNumThreadsToWaitFor = uiNumThreadsToWaitFor;
                }

            private:
                std::mutex m_mtxBarrier;
                std::condition_variable m_cvAllThreadsReachedBarrier;
                std::size_t m_uiNumThreadsToWaitFor;
            };
        }
    }
}
