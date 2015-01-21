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

#include <alpaka/traits/stream.hpp>
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/cuda/Common.hpp>
#include <alpaka/cuda/AccCudaFwd.hpp>   // AccCuda

// forward declarations
namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            class EventCuda;
        }
    }
}

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator stream.
            //#############################################################################
            class StreamCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST Stream()
                {
                    ALPAKA_CUDA_CHECK(cudaStreamCreate(
                        &m_cudaStream);
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST Stream(Stream const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST Stream(Stream &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST Stream & operator=(Stream const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(Stream const & rhs) const
                {
                    return m_cudaStream == rhs.m_cudaStream;
                }
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(Stream const & rhs) const
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~Stream() noexcept
                {
                    ALPAKA_CUDA_CHECK(cudaStreamDestroy(m_cudaStream));
                }

                cudaStream_t m_cudaStream;
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator stream accelerator type trait specialization.
            //#############################################################################
            template<>
            struct GetAcc<
                cuda::detail::StreamCuda>
            {
                using type = AccCuda;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA accelerator stream type trait specialization.
            //#############################################################################
            template<>
            class GetStream<
                AccCuda>
            {
                using type = cuda::detail::StreamCuda;
            };

            //#############################################################################
            //! The CUDA accelerator stream test trait specialization.
            //#############################################################################
            template<>
            struct StreamTest<
                cuda::detail::StreamCuda>
            {
                static ALPAKA_FCT_HOST bool streamTest(
                    cuda::detail::StreamCuda const & stream)
                {
                    auto const ret(cudaStreamQuery(stream.m_cudaStream));
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
                        throw std::runtime_error(("Unexpected return value '" + std::string(cudaGetErrorString(ret)) + "'from cudaStreamQuery!").c_str());
                    }
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CUDA accelerator stream thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                cuda::detail::StreamCuda>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    cuda::detail::StreamCuda const & stream)
                {
                    ALPAKA_CUDA_CHECK(cudaStreamSynchronize(
                        stream.m_cudaStream));
                }
            };

            //#############################################################################
            //! The CUDA accelerator stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                cuda::detail::StreamCuda,
                cuda::detail::EventCuda>
            {
                ALPAKA_FCT_HOST static void waiterWaitFor(
                    cuda::detail::StreamCuda const & stream,
                    cuda::detail::EventCuda const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaStreamWaitEvent(
                        stream.m_cudaStream,
                        event.m_cudaEvent,
                        0));
                }
            };
        }
    }
}
