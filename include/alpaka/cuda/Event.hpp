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

#include <alpaka/cuda/Common.hpp>
#include <alpaka/cuda/AccCudaFwd.hpp>   // AccCuda

#include <alpaka/traits/Event.hpp>

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
            using Acc = AccCuda;

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
                    &m_cudaEvent,
                    (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event(Event const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event(Event &&) = default;
            //-----------------------------------------------------------------------------
            //! Assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Event & operator=(Event const &) = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST virtual ~Event() noexcept
            {
                ALPAKA_CUDA_CHECK(cudaEventDestroy(m_cudaEvent));
            }

            cudaEvent_t m_cudaEvent;
        };
    }

    namespace traits
    {
        namespace event
        {
            //#############################################################################
            //! The CUDA accelerator event enqueue trait specialization.
            //#############################################################################
            template<>
            struct DefaultStreamEnqueueEvent
            <
                alpaka::event::Event<AccCuda>,
                stream::Stream<AccCuda>
            >
            {
                static ALPAKA_FCT_HOST void defaultStreamEnqueueEvent(
                    alpaka::event::Event<AccCuda> const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaEventRecord(
                        event.m_cudaEvent,
                        nullptr));
                }
            };

            //#############################################################################
            //! The CUDA accelerator event enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueueEvent
            <
                alpaka::event::Event<AccCuda>,
                stream::Stream<AccCuda>
            >
            {
                static ALPAKA_FCT_HOST void streamEnqueueEvent(
                    alpaka::event::Event<AccCuda> const & event, 
                    stream::Stream<AccCuda> const * stream)
                {
                    ALPAKA_CUDA_CHECK(cudaEventRecord(
                        event.m_cudaEvent,
                        &stream->m_cudaStream));
                }
            };

            //#############################################################################
            //! The CUDA accelerator thread event wait trait specialization.
            //#############################################################################
            template<>
            struct ThreadWaitEvent
            <
                alpaka::event::Event<AccCuda>
            >
            {
                static ALPAKA_FCT_HOST void threadWaitEvent(
                    alpaka::event::Event<AccCuda> const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaEventSynchronize(event.m_cudaEvent));
                }
            };

            //#############################################################################
            //! The CUDA accelerator event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest
            <
                alpaka::event::Event<AccCuda>
            >
            {
                static ALPAKA_FCT_HOST bool eventTest(
                    alpaka::event::Event<AccCuda> const & event)
                {
                    auto const ret(cudaEventQuery(event.m_cudaEvent));
                    if(ret == cudaSuccess)
                    {
                        return true;
                    }
                    else if(ret == cudaErrorNotReady)
                    {
                        return false;
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
