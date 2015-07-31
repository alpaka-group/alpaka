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

#include <alpaka/dev/Traits.hpp>            // dev::getDev
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::DimInt<N>
#include <alpaka/extent/Traits.hpp>         // view::getXXX
#include <alpaka/mem/view/Traits.hpp>       // view::Set
#include <alpaka/stream/Traits.hpp>         // stream::Enqueue

#include <alpaka/core/Cuda.hpp>             // cudaMemset, ...

#include <cassert>                          // assert

namespace alpaka
{
    namespace dev
    {
        class DevCudaRt;
    }
}

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for Set.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        namespace view
        {
            namespace cuda
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CUDA memory copy trait.
                    //#############################################################################
                    template<
                        typename TDim,
                        typename TBuf,
                        typename TExtents>
                    struct TaskSet
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        TaskSet(
                            TBuf & buf,
                            std::uint8_t const & byte,
                            TExtents const & extents) :
                                m_buf(buf),
                                m_byte(byte),
                                m_extents(extents),
                                m_iDevice(dev::getDev(buf).m_iDevice)
                        {
                            static_assert(
                                dim::Dim<TBuf>::value == 1u,
                                "The destination buffer is required to be 1-dimensional for this specialization!");
                            static_assert(
                                dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                                "The destination buffer and the extents are required to have the same dimensionality!");
                        }

                        TBuf & m_buf;
                        std::uint8_t const m_byte;
                        TExtents const m_extents;
                        std::int32_t const m_iDevice;
                    };
                }
            }
            namespace traits
            {
                //#############################################################################
                //! The CPU device memory set trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct TaskSet<
                    TDim,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto taskSet(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents)
                    -> mem::view::cuda::detail::TaskSet<
                        TDim,
                        TBuf,
                        TExtents>
                    {
                        return
                            mem::view::cuda::detail::TaskSet<
                                TDim,
                                TBuf,
                                TExtents>(
                                    buf,
                                    byte,
                                    extents);
                    }
                };
            }
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA async device stream 1D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TBuf,
                typename TExtents>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskSet<dim::DimInt<1u>, TBuf, TExtents>>
            {
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskSet<dim::DimInt<1u>, TBuf, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TBuf>::value == 1u,
                        "The destination buffer is required to be 1-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extents(task.m_extents);
                    auto const & iDevice(task.m_iDevice);

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(mem::view::Elem<TBuf>));
                    auto const uiDstWidth(extent::getWidth(buf));
                    auto const pDstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(uiExtentWidth <= uiDstWidth);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemsetAsync(
                            pDstNativePtr,
                            static_cast<int>(byte),
                            uiExtentWidthBytes,
                            stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 1D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TBuf,
                typename TExtents>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskSet<dim::DimInt<1u>, TBuf, TExtents>>
            {
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    mem::view::cuda::detail::TaskSet<dim::DimInt<1u>, TBuf, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TBuf>::value == 1u,
                        "The destination buffer is required to be 1-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extents(task.m_extents);
                    auto const & iDevice(task.m_iDevice);

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(mem::view::Elem<TBuf>));
                    auto const uiDstWidth(extent::getWidth(buf));
                    auto const pDstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(uiExtentWidth <= uiDstWidth);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset(
                            pDstNativePtr,
                            static_cast<int>(byte),
                            uiExtentWidthBytes));
                }
            };
            //#############################################################################
            //! The CUDA async device stream 2D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TBuf,
                typename TExtents>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskSet<dim::DimInt<2u>, TBuf, TExtents>>
            {
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskSet<dim::DimInt<2u>, TBuf, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TBuf>::value == 2u,
                        "The destination buffer is required to be 2-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extents(task.m_extents);
                    auto const & iDevice(task.m_iDevice);

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(mem::view::Elem<TBuf>));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiDstWidth(extent::getWidth(buf));
                    auto const uiDstHeight(extent::getHeight(buf));
                    auto const uiDstPitchBytes(mem::view::getPitchBytes<std::integral_constant<std::size_t, dim::Dim<TBuf>::value - 1u>>(buf));
                    auto const pDstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset2DAsync(
                            pDstNativePtr,
                            uiDstPitchBytes,
                            static_cast<int>(byte),
                            uiExtentWidthBytes,
                            uiExtentHeight,
                            stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 2D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TBuf,
                typename TExtents>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskSet<dim::DimInt<2u>, TBuf, TExtents>>
            {
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    mem::view::cuda::detail::TaskSet<dim::DimInt<2u>, TBuf, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TBuf>::value == 2u,
                        "The destination buffer is required to be 2-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extents(task.m_extents);
                    auto const & iDevice(task.m_iDevice);

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(mem::view::Elem<TBuf>));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiDstWidth(extent::getWidth(buf));
                    auto const uiDstHeight(extent::getHeight(buf));
                    auto const uiDstPitchBytes(mem::view::getPitchBytes<std::integral_constant<std::size_t, dim::Dim<TBuf>::value - 1u>>(buf));
                    auto const pDstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset2D(
                            pDstNativePtr,
                            uiDstPitchBytes,
                            static_cast<int>(byte),
                            uiExtentWidthBytes,
                            uiExtentHeight));
                }
            };
            //#############################################################################
            //! The CUDA async device stream 3D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TBuf,
                typename TExtents>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskSet<dim::DimInt<3u>, TBuf, TExtents>>
            {
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskSet<dim::DimInt<3u>, TBuf, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TBuf>::value == 3u,
                        "The destination buffer is required to be 3-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extents(task.m_extents);
                    auto const & iDevice(task.m_iDevice);

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiExtentDepth(extent::getDepth(extents));
                    auto const uiDstWidth(extent::getWidth(buf));
                    auto const uiDstHeight(extent::getHeight(buf));
                    auto const uiDstDepth(extent::getDepth(buf));
                    auto const uiDstPitchBytes(mem::view::getPitchBytes<std::integral_constant<std::size_t, dim::Dim<TBuf>::value - 1u>>(buf));
                    auto const pDstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            pDstNativePtr,
                            uiDstPitchBytes,
                            uiDstWidth,
                            uiDstHeight));

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            uiExtentWidth,
                            uiExtentHeight,
                            uiExtentDepth));

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset3DAsync(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal,
                            stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 3D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TBuf,
                typename TExtents>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskSet<dim::DimInt<3u>, TBuf, TExtents>>
            {
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    mem::view::cuda::detail::TaskSet<dim::DimInt<3u>, TBuf, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TBuf>::value == 3u,
                        "The destination buffer is required to be 3-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extents(task.m_extents);
                    auto const & iDevice(task.m_iDevice);

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiExtentDepth(extent::getDepth(extents));
                    auto const uiDstWidth(extent::getWidth(buf));
                    auto const uiDstHeight(extent::getHeight(buf));
                    auto const uiDstDepth(extent::getDepth(buf));
                    auto const uiDstPitchBytes(mem::view::getPitchBytes<std::integral_constant<std::size_t, dim::Dim<TBuf>::value - 1u>>(buf));
                    auto const pDstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            pDstNativePtr,
                            uiDstPitchBytes,
                            uiDstWidth,
                            uiDstHeight));

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            uiExtentWidth,
                            uiExtentHeight,
                            uiExtentDepth));

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset3D(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal));
                }
            };
        }
    }
}
