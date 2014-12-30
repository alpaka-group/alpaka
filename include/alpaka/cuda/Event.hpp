/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/cuda/Common.hpp>
#include <alpaka/cuda/AccCudaFwd.hpp>   // AccCuda

#include <alpaka/interfaces/Event.hpp>

namespace alpaka
{
    namespace event
    {
        //#############################################################################
        //! The CUDA accelerator event.
        //#############################################################################
        template<>
        class Event<AccCuda>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event(bool bBusyWait = true)
            {
                // Creates an event object with the specified flags. Valid flags include:
                //  cudaEventDefault: Default event creation flag.
                //  cudaEventBlockingSync : Specifies that event should use blocking synchronization.A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                //  cudaEventDisableTiming : Specifies that the created event does not need to record timing data.Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                ALPAKA_CUDA_CHECK(cudaEventCreateWithFlags(
                    &m_Event,
                    (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
            }
            //-----------------------------------------------------------------------------
            //! Copy-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event(Event const &) = default;
            //-----------------------------------------------------------------------------
            //! Move-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event(Event &&) = default;
            //-----------------------------------------------------------------------------
            //! Assignment-operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event & operator=(Event const &) = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~Event() noexcept
            {
                ALPAKA_CUDA_CHECK(cudaEventDestroy(m_Event));
            }

            cudaEvent_t m_Event;
        };

        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator event enqueuer.
            //#############################################################################
            template<>
            struct EventEnqueue<Event<AccCuda>>
            {
                ALPAKA_FCT_HOST EventEnqueue(Event<AccCuda> const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaEventRecord(
                        event.m_Event,
                        0));
                }
            };

            //#############################################################################
            //! The CUDA accelerator event waiter.
            //#############################################################################
            template<>
            struct EventWait<Event<AccCuda>>
            {
                ALPAKA_FCT_HOST EventWait(Event<AccCuda> const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaEventSynchronize(event.m_Event));
                }
            };

            //#############################################################################
            //! The CUDA accelerator event tester.
            //#############################################################################
            template<>
            struct EventTest<Event<AccCuda>>
            {
                ALPAKA_FCT_HOST EventTest(Event<AccCuda> const & event, bool & bTest)
                {
                    auto const ret(cudaEventQuery(event.m_Event));
                    if(ret == cudaSuccess)
                    {
                        bTest = true;
                    }
                    else if(ret == cudaErrorNotReady)
                    {
                        bTest = false;
                    }
                    else
                    {
                        throw std::runtime_error(("Unexpected return value '" + std::string(cudaGetErrorString(ret)) + "'from cudaEventQuery!").c_str());
                    }
                }
            };
        }
    }
}
