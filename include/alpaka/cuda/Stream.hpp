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

#include <alpaka/traits/Stream.hpp>
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor, WaiterWaitFor
#include <alpaka/traits/Acc.hpp>        // AccType

#include <alpaka/cuda/AccCudaFwd.hpp>   // AccCuda
#include <alpaka/cuda/Common.hpp>       // ALPAKA_CUDA_CHECK

#include <memory>                       // std::shared_ptr

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
                ALPAKA_FCT_HOST StreamCuda() :
                    m_spCudaStream(new cudaStream_t, &StreamCuda::destroyStream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_CHECK(cudaStreamCreate(
                        m_spCudaStream.get()));
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCuda(StreamCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCuda(StreamCuda &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCuda & operator=(StreamCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(StreamCuda const & rhs) const
                {
                    return (*m_spCudaStream.get()) == (*rhs.m_spCudaStream.get());
                }
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(StreamCuda const & rhs) const
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST /*virtual*/ ~StreamCuda() noexcept = default;
                
            private:
                //-----------------------------------------------------------------------------
                //! Destroys the shared stream.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void destroyStream(
                    cudaStream_t * stream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately 
                    // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                    // -> No need to synchronize here.
                    ALPAKA_CUDA_CHECK(cudaStreamDestroy(*stream));
                }

            public:
                std::shared_ptr<cudaStream_t> m_spCudaStream;
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
            struct AccType<
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
            struct StreamType<
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
                ALPAKA_FCT_HOST static bool streamTest(
                    cuda::detail::StreamCuda const & stream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const ret(
                        cudaStreamQuery(
                            *stream.m_spCudaStream.get()));
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
                        throw std::runtime_error(("Unexpected return value '" + std::string(cudaGetErrorString(ret)) + "'from cudaStreamQuery!"));
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
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_CHECK(cudaStreamSynchronize(
                        *stream.m_spCudaStream.get()));
                }
            };
        }
    }
}
