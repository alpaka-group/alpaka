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

#include <alpaka/cuda/Stream.hpp>       // StreamCuda
#include <alpaka/cuda/Event.hpp>        // EventCuda
#include <alpaka/cuda/Common.hpp>       // ALPAKA_CUDA_CHECK

namespace alpaka
{
    namespace traits
    {
        namespace event
        {
            //#############################################################################
            //! The CUDA accelerator event enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueueEvent<
                cuda::detail::EventCuda,
                cuda::detail::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto streamEnqueueEvent(
                    cuda::detail::EventCuda const & event,
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_CHECK(cudaEventRecord(
                        *event.m_spCudaEvent.get(),
                        *stream.m_spCudaStream.get()));
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CUDA accelerator stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                cuda::detail::StreamCuda,
                cuda::detail::EventCuda>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    cuda::detail::StreamCuda const & stream,
                    cuda::detail::EventCuda const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_CHECK(cudaStreamWaitEvent(
                        *stream.m_spCudaStream.get(),
                        *event.m_spCudaEvent.get(),
                        0));
                }
            };
        }
    }
}