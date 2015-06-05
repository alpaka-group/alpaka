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

#include <alpaka/devs/cuda/Dev.hpp>         // DevCuda

#include <alpaka/traits/Stream.hpp>
#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor, WaiterWaitFor
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Dev.hpp>            // GetDev

#include <alpaka/core/Cuda.hpp>             // ALPAKA_CUDA_RT_CHECK

#include <stdexcept>                        // std::runtime_error
#include <memory>                           // std::shared_ptr
#include <functional>                       // std::bind

namespace alpaka
{
    namespace devs
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA device stream implementation.
                //#############################################################################
                class StreamCudaImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    StreamCudaImpl(
                        DevCuda const & dev) :
                            m_Dev(dev),
                            m_CudaStream()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_Dev.m_iDevice));
                        // - cudaStreamDefault: Default stream creation flag.
                        // - cudaStreamNonBlocking: Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream),
                        //   and that the created stream should perform no implicit synchronization with stream 0.
                        // Create the stream on the current device.
                        // NOTE: cudaStreamNonBlocking is required to match the semantic implemented in the CPU stream.
                        // It would be too much work to implement implicit default stream synchronization on CPU.
                        ALPAKA_CUDA_RT_CHECK(cudaStreamCreateWithFlags(
                            &m_CudaStream,
                            cudaStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCudaImpl(StreamCudaImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST StreamCudaImpl(StreamCudaImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(StreamCudaImpl const &) -> StreamCudaImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(StreamCudaImpl &&) -> StreamCudaImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ~StreamCudaImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaStreamDestroy required?
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_Dev.m_iDevice));
                        // In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately
                        // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(cudaStreamDestroy(
                            m_CudaStream));
                    }

                public:
                    DevCuda const m_Dev;        //!< The device this stream is bound to.
                    cudaStream_t m_CudaStream;
                };
            }

            //#############################################################################
            //! The CUDA device stream.
            //#############################################################################
            class StreamCuda final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCuda(
                    DevCuda & dev) :
                    m_spStreamCudaImpl(std::make_shared<detail::StreamCudaImpl>(dev))
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCuda(StreamCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamCuda(StreamCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(StreamCuda const &) -> StreamCuda & = default;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(StreamCuda &&) -> StreamCuda & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(StreamCuda const & rhs) const
                -> bool
                {
                    return (m_spStreamCudaImpl->m_CudaStream == rhs.m_spStreamCudaImpl->m_CudaStream);
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
                ALPAKA_FCT_HOST ~StreamCuda() noexcept = default;

            public:
                std::shared_ptr<detail::StreamCudaImpl> m_spStreamCudaImpl;
            };
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The CUDA device stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                devs::cuda::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cuda::StreamCuda const & stream)
                -> devs::cuda::DevCuda
                {
                    return stream.m_spStreamCudaImpl->m_Dev;
                }
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA device stream stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                devs::cuda::StreamCuda>
            {
                using type = devs::cuda::StreamCuda;
            };

            //#############################################################################
            //! The CUDA device stream test trait specialization.
            //#############################################################################
            template<>
            struct StreamTest<
                devs::cuda::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto streamTest(
                    devs::cuda::StreamCuda const & stream)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for streams on non current device.
                    auto const ret(
                        cudaStreamQuery(
                            stream.m_spStreamCudaImpl->m_CudaStream));
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
            //! The CUDA device stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                devs::cuda::StreamCuda>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    devs::cuda::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for streams on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaStreamSynchronize(
                        stream.m_spStreamCudaImpl->m_CudaStream));
                }
            };
        }
    }
}
