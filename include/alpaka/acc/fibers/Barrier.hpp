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

#include <alpaka/core/Fibers.hpp>

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_ACC_NO_CUDA

#include <mutex>                    // std::unique_lock

namespace alpaka
{
    namespace acc
    {
        namespace fibers
        {
            //#############################################################################
            //! A barrier.
            // NOTE: We do not use the boost::fibers::barrier because it does not support simple resetting.
            //#############################################################################
            template<
                typename TSize>
            class FiberBarrier
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA explicit FiberBarrier(
                    TSize const & uiNumFibersToWaitFor = 0) :
                    m_uiNumFibersToWaitFor{uiNumFibersToWaitFor}
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA FiberBarrier(FiberBarrier const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA FiberBarrier(FiberBarrier &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(FiberBarrier const &) -> FiberBarrier & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(FiberBarrier &&) -> FiberBarrier & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~FiberBarrier() = default;

                //-----------------------------------------------------------------------------
                //! Waits for all the other fibers to reach the barrier.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto wait()
                -> void
                {
                    std::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
                    if(--m_uiNumFibersToWaitFor == 0)
                    {
                        m_cvAllFibersReachedBarrier.notify_all();
                    }
                    else
                    {
                        m_cvAllFibersReachedBarrier.wait(lock, [this] { return m_uiNumFibersToWaitFor == 0; });
                    }
                }

                //-----------------------------------------------------------------------------
                //! \return The number of fibers to wait for.
                //! NOTE: The value almost always is invalid the time you get it.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto getNumFibersToWaitFor() const
                -> TSize
                {
                    return m_uiNumFibersToWaitFor;
                }

                //-----------------------------------------------------------------------------
                //! Resets the number of fibers to wait for to the given number.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto reset(
                    TSize const & uiNumFibersToWaitFor)
                -> void
                {
                    // A lock is not required in the fiber implementation.
                    //boost::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
                    m_uiNumFibersToWaitFor = uiNumFibersToWaitFor;
                }

            private:
                boost::fibers::mutex m_mtxBarrier;
                boost::fibers::condition_variable m_cvAllFibersReachedBarrier;
                TSize m_uiNumFibersToWaitFor;
            };
        }
    }
}
