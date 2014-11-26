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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC

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
                ALPAKA_FCT_HOST explicit ThreadBarrier(std::size_t const uiNumThreadsToWaitFor = 0) :
                    m_uiNumThreadsToWaitFor{uiNumThreadsToWaitFor}
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ThreadBarrier(ThreadBarrier const & other) :
                    m_uiNumThreadsToWaitFor(other.m_uiNumThreadsToWaitFor)
                {}
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ThreadBarrier(ThreadBarrier &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ThreadBarrier & operator=(ThreadBarrier const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~ThreadBarrier() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Waits for all the other threads to reach the barrier.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void wait()
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
                ALPAKA_FCT_HOST std::size_t getNumThreadsToWaitFor() const
                {
                    return m_uiNumThreadsToWaitFor;
                }

                //-----------------------------------------------------------------------------
                //! Resets the number of threads to wait for to the given number.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void reset(std::size_t const uiNumThreadsToWaitFor)
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
