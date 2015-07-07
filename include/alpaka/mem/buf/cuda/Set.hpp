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

#include <alpaka/dev/DevCudaRt.hpp>         // DevCudaRt
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::DimInt<N>
#include <alpaka/extent/Traits.hpp>         // view::getXXX
#include <alpaka/mem/view/Traits.hpp>       // view::Set
#include <alpaka/stream/StreamCudaRt.hpp>   // StreamCudaRt

#include <alpaka/core/Cuda.hpp>             // cudaMemset, ...

#include <cassert>                          // assert

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for Set.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The CUDA 1D memory set trait specialization.
                //#############################################################################
                template<>
                struct Set<
                    dim::DimInt<1u>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::Dim<TBuf>::value == 1u,
                            "The destination buffer is required to be 1-dimensional for this specialization!");
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        assert(uiExtentWidth <= uiDstWidth);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev::getDev(buf).m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemset(
                                reinterpret_cast<void *>(mem::view::getPtrNative(buf)),
                                static_cast<int>(byte),
                                uiExtentWidth * sizeof(mem::view::Elem<TBuf>)));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBuf,
                        typename TExtents>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents,
                        stream::StreamCudaRt const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::Dim<TBuf>::value == 1u,
                            "The destination buffer is required to be 1-dimensional for this specialization!");
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        assert(uiExtentWidth <= uiDstWidth);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev::getDev(buf).m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemsetAsync(
                                reinterpret_cast<void *>(mem::view::getPtrNative(buf)),
                                static_cast<int>(byte),
                                uiExtentWidth * sizeof(mem::view::Elem<TBuf>),
                                stream.m_spStreamCudaImpl->m_CudaStream));
                    }
                };
                //#############################################################################
                //! The CUDA 2D memory set trait specialization.
                //#############################################################################
                template<>
                struct Set<
                    dim::DimInt<2u>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::Dim<TBuf>::value == 2u,
                            "The destination buffer is required to be 2-dimensional for this specialization!");
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiExtentHeight(extent::getHeight(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        auto const uiDstHeight(extent::getHeight(buf));
                        auto const uiDstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBuf>::value - 1u>(buf));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev::getDev(buf).m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemset2D(
                                reinterpret_cast<void *>(mem::view::getPtrNative(buf)),
                                uiDstPitchBytes,
                                static_cast<int>(byte),
                                uiExtentWidth * sizeof(mem::view::Elem<TBuf>),
                                uiExtentHeight));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents,
                        stream::StreamCudaRt const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::Dim<TBuf>::value == 2u,
                            "The destination buffer is required to be 2-dimensional for this specialization!");
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiExtentHeight(extent::getHeight(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        auto const uiDstHeight(extent::getHeight(buf));
                        auto const uiDstPitchBytes(mem::view::getPitchBytes<std::integral_constant<std::size_t, dim::Dim<TBuf>::value - 1u>>(buf));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev::getDev(buf).m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemset2DAsync(
                                reinterpret_cast<void *>(mem::view::getPtrNative(buf)),
                                uiDstPitchBytes,
                                static_cast<int>(byte),
                                uiExtentWidth * sizeof(mem::view::Elem<TBuf>),
                                uiExtentHeight,
                                stream.m_spStreamCudaImpl->m_CudaStream));
                    }
                };
                //#############################################################################
                //! The CUDA 3D memory set trait specialization.
                //#############################################################################
                template<>
                struct Set<
                    dim::DimInt<3u>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::Dim<TBuf>::value == 3u,
                            "The destination buffer is required to be 3-dimensional for this specialization!");
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiExtentHeight(extent::getHeight(extents));
                        auto const uiExtentDepth(extent::getDepth(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        auto const uiDstHeight(extent::getHeight(buf));
                        auto const uiDstDepth(extent::getDepth(buf));
                        auto const uiDstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBuf>::value - 1u>(buf));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentDepth <= uiDstDepth);

                        // Fill CUDA parameter structures.
                        cudaPitchedPtr const cudaPitchedPtrVal(
                            make_cudaPitchedPtr(
                                reinterpret_cast<void *>(mem::view::getPtrNative(buf)),
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight));

                        cudaExtent const cudaExtentVal(
                            make_cudaExtent(
                                uiExtentWidth,
                                uiExtentHeight,
                                uiExtentDepth));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev::getDev(buf).m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemset3D(
                                cudaPitchedPtrVal,
                                static_cast<int>(byte),
                                cudaExtentVal));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents,
                        stream::StreamCudaRt const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::Dim<TBuf>::value == 3u,
                            "The destination buffer is required to be 3-dimensional for this specialization!");
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiExtentHeight(extent::getHeight(extents));
                        auto const uiExtentDepth(extent::getDepth(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        auto const uiDstHeight(extent::getHeight(buf));
                        auto const uiDstDepth(extent::getDepth(buf));
                        auto const uiDstPitchBytes(mem::view::getPitchBytes<std::integral_constant<std::size_t, dim::Dim<TBuf>::value - 1u>>(buf));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentDepth <= uiDstDepth);

                        // Fill CUDA parameter structures.
                        cudaPitchedPtr const cudaPitchedPtrVal(
                            make_cudaPitchedPtr(
                                reinterpret_cast<void *>(mem::view::getPtrNative(buf)),
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight));

                        cudaExtent const cudaExtentVal(
                            make_cudaExtent(
                                uiExtentWidth,
                                uiExtentHeight,
                                uiExtentDepth));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev::getDev(buf).m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemset3DAsync(
                                cudaPitchedPtrVal,
                                static_cast<int>(byte),
                                cudaExtentVal,
                                stream.m_spStreamCudaImpl->m_CudaStream));
                    }
                };
            }
        }
    }
}
