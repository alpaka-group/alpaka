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
#include <alpaka/cuda/Stream.hpp>       // StreamCuda

#include <alpaka/traits/Event.hpp>
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor
#include <alpaka/traits/Acc.hpp>        // GetAcc

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator event.
            //#############################################################################
            class EventCuda
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
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator event accelerator type trait specialization.
            //#############################################################################
            template<>
            struct GetAcc<
                cuda::detail::EventCuda>
            {
                using type = AccCuda;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CUDA accelerator event type trait specialization.
            //#############################################################################
            template<>
            class GetEvent<
                AccCuda>
            {
                using type = alpaka::cuda::detail::EventCuda;
            };

            //#############################################################################
            //! The CUDA accelerator event enqueue trait specialization.
            //#############################################################################
            template<>
            struct DefaultStreamEnqueueEvent<
                cuda::detail::EventCuda,
                cuda::detail::StreamCuda>
            {
                static ALPAKA_FCT_HOST void defaultStreamEnqueueEvent(
                    cuda::detail::EventCuda const & event)
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
            struct StreamEnqueueEvent<
                cuda::detail::EventCuda,
                cuda::detail::StreamCuda>
            {
                static ALPAKA_FCT_HOST void streamEnqueueEvent(
                    cuda::detail::EventCuda const & event,
                    cuda::detail::StreamCuda const * stream)
                {
                    ALPAKA_CUDA_CHECK(cudaEventRecord(
                        event.m_cudaEvent,
                        &stream->m_cudaStream));
                }
            };

            //#############################################################################
            //! The CUDA accelerator event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                cuda::detail::EventCuda>
            {
                static ALPAKA_FCT_HOST bool eventTest(
                    cuda::detail::EventCuda const & event)
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

        namespace wait
        {
            //#############################################################################
            //! The CUDA accelerator event thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                cuda::detail::EventCuda>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    cuda::detail::EventCuda const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaEventSynchronize(event.m_cudaEvent));
                }
            };
        }
    }
}
