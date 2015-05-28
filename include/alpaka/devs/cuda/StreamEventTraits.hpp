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

#include <alpaka/devs/cuda/Stream.hpp>  // StreamCuda
#include <alpaka/devs/cuda/Event.hpp>   // EventCuda

#include <alpaka/core/Cuda.hpp>         // ALPAKA_CUDA_RT_CHECK

namespace alpaka
{
    namespace traits
    {
        namespace stream
        {
            //#############################################################################
            //! The CUDA device stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct StreamEnqueue<
                devs::cuda::StreamCuda,
                devs::cuda::EventCuda>
            {
                ALPAKA_FCT_HOST static auto streamEnqueue(
                    devs::cuda::StreamCuda & stream,
                    devs::cuda::EventCuda & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventCudaImpl->m_CudaEvent,
                        stream.m_spStreamCudaImpl->m_CudaStream));
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CUDA device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                devs::cuda::StreamCuda,
                devs::cuda::EventCuda>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    devs::cuda::StreamCuda & stream,
                    devs::cuda::EventCuda const & event)
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
            //! The CUDA device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                devs::cuda::DevCuda,
                devs::cuda::EventCuda>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    devs::cuda::DevCuda & dev,
                    devs::cuda::EventCuda const & event)
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