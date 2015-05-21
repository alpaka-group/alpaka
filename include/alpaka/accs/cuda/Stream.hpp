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

#include <alpaka/accs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/accs/cuda/Common.hpp>      // ALPAKA_CUDA_RT_CHECK

#include <alpaka/traits/Stream.hpp>
#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor, WaiterWaitFor
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Dev.hpp>            // GetDev

#include <stdexcept>                        // std::runtime_error
#include <memory>                           // std::shared_ptr
#include <functional>                       // std::bind

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                template<
                    typename TDim>
                class AccGpuCuda;

                //#############################################################################
                //! The GPU CUDA accelerator stream.
                //#############################################################################
                class StreamCuda
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCuda(
                        DevCuda & dev) :
                        m_spCudaStream(
                            new cudaStream_t,
                            std::bind(&StreamCuda::destroyStream, std::placeholders::_1, std::ref(m_Dev))),
                        m_Dev(dev)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_Dev.m_iDevice));
                        // Create the stream on the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaStreamCreate(
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
                    ALPAKA_FCT_HOST auto operator=(StreamCuda const &) -> StreamCuda & = default;
                    //-----------------------------------------------------------------------------
                    //! Equality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator==(StreamCuda const & rhs) const
                    -> bool
                    {
                        return (*m_spCudaStream.get()) == (*rhs.m_spCudaStream.get());
                    }
                    //-----------------------------------------------------------------------------
                    //! Equality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator!=(StreamCuda const & rhs) const
                    -> bool
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
                    ALPAKA_FCT_HOST static auto destroyStream(
                        cudaStream_t * stream,
                        DevCuda const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaStreamDestroy required?
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev.m_iDevice));
                        // In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately
                        // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(cudaStreamDestroy(
                            *stream));
                    }

                public:
                    std::shared_ptr<cudaStream_t> m_spCudaStream;
                    DevCuda m_Dev;
                };
            }
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The GPU CUDA accelerator stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                accs::cuda::detail::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::cuda::detail::StreamCuda const & stream)
                -> accs::cuda::detail::DevCuda
                {
                    return stream.m_Dev;
                }
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The GPU CUDA accelerator stream stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::cuda::detail::StreamCuda>
            {
                using type = accs::cuda::detail::StreamCuda;
            };

            //#############################################################################
            //! The GPU CUDA accelerator stream test trait specialization.
            //#############################################################################
            template<>
            struct StreamTest<
                accs::cuda::detail::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto streamTest(
                    accs::cuda::detail::StreamCuda const & stream)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for streams on non current device.
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
            //! The GPU CUDA accelerator stream thread wait trait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                accs::cuda::detail::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    accs::cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for streams on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaStreamSynchronize(
                        *stream.m_spCudaStream.get()));
                }
            };
        }
    }
}
