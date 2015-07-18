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

#include <alpaka/event/EventCudaRt.hpp>     // EventCuda
#include <alpaka/stream/StreamCudaRt.hpp>   // stream::StreamCudaRt

#include <alpaka/core/Cuda.hpp>             // ALPAKA_CUDA_RT_CHECK

namespace alpaka
{
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                stream::StreamCudaRt>
            {
                using type = event::EventCudaRt;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueue<
                stream::StreamCudaRt,
                event::EventCudaRt>
            {
                ALPAKA_FN_HOST static auto streamEnqueue(
                    stream::StreamCudaRt & stream,
                    event::EventCudaRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventCudaImpl->m_CudaEvent,
                        stream.m_spStreamCudaImpl->m_CudaStream));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCudaRt,
                event::EventCudaRt>
            {
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCudaRt & stream,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        stream.m_spStreamCudaImpl->m_CudaStream,
                        event.m_spEventCudaImpl->m_CudaEvent,
                        0));
                }
            };

            //#############################################################################
            //! The CUDA RT device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                dev::DevCudaRt,
                event::EventCudaRt>
            {
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCudaRt & dev,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        0,
                        event.m_spEventCudaImpl->m_CudaEvent,
                        0));
                }
            };
        }
    }
}
