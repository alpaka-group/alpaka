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
#include <alpaka/accs/cuda/Stream.hpp>      // StreamCuda
#include <alpaka/accs/cuda/Common.hpp>

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>

#include <alpaka/traits/Mem.hpp>            // traits::Copy
#include <alpaka/traits/Extent.hpp>         // traits::getXXX

#include <cassert>                          // assert

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA memory copy trait.
                //#############################################################################
                template<
                    typename TDim>
                struct MemCopyCuda;
                //#############################################################################
                //! The CUDA 1D memory copy trait specialization.
                //#############################################################################
                template<>
                struct MemCopyCuda<
                    dim::Dim1>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCuda(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind,
                        std::int32_t const & iDev)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentWidth <= uiSrcWidth);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            iDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                reinterpret_cast<void const *>(mem::getPtrNative(bufSrc)),
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                p_cudaMemcpyKind));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " sw: " << uiSrcWidth
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << std::endl;
#endif
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCuda(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind,
                        std::int32_t const & iDev,
                        cuda::detail::StreamCuda const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentWidth <= uiSrcWidth);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            iDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyAsync(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                reinterpret_cast<void const *>(mem::getPtrNative(bufSrc)),
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                p_cudaMemcpyKind,
                                *stream.m_spCudaStream.get()));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " sw: " << uiSrcWidth
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << std::endl;
#endif
                    }
                };
                //#############################################################################
                //! The CUDA 2D memory copy trait specialization.
                //#############################################################################
                template<>
                struct MemCopyCuda<
                    dim::Dim2>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCuda(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind,
                        std::int32_t const & iDev)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiExtentHeight(extent::getHeight<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiDstHeight(extent::getHeight<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        auto const uiSrcHeight(extent::getHeight<UInt>(bufSrc));
                        auto const uiDstPitchBytes(mem::getPitchBytes<dim::DimT<TBufDst>::value - 1u, UInt>(bufDst));
                        auto const uiSrcPitchBytes(mem::getPitchBytes<dim::DimT<TBufSrc>::value - 1u, UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentWidth <= uiSrcWidth);
                        assert(uiExtentHeight <= uiSrcHeight);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            iDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2D(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                uiDstPitchBytes,
                                reinterpret_cast<void const *>(mem::getPtrNative(bufSrc)),
                                uiSrcPitchBytes,
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                uiExtentHeight,
                                p_cudaMemcpyKind));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " eh: " << uiExtentHeight
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dh: " << uiDstHeight
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " dpitchb: " << uiDstPitchBytes
                            << " sw: " << uiSrcWidth
                            << " sh: " << uiSrcHeight
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << " spitchb: " << uiSrcPitchBytes
                            << std::endl;
#endif
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCuda(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind,
                        std::int32_t const & iDev,
                        cuda::detail::StreamCuda const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiExtentHeight(extent::getHeight<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiDstHeight(extent::getHeight<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        auto const uiSrcHeight(extent::getHeight<UInt>(bufSrc));
                        auto const uiDstPitchBytes(mem::getPitchBytes<dim::DimT<TBufDst>::value - 1u, UInt>(bufDst));
                        auto const uiSrcPitchBytes(mem::getPitchBytes<dim::DimT<TBufSrc>::value - 1u, UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentWidth <= uiSrcWidth);
                        assert(uiExtentHeight <= uiSrcHeight);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            iDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2DAsync(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                uiDstPitchBytes,
                                reinterpret_cast<void const *>(mem::getPtrNative(bufSrc)),
                                uiSrcPitchBytes,
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                uiExtentHeight,
                                p_cudaMemcpyKind,
                                *stream.m_spCudaStream.get()));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " eh: " << uiExtentHeight
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dh: " << uiDstHeight
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " dpitchb: " << uiDstPitchBytes
                            << " sw: " << uiSrcWidth
                            << " sh: " << uiSrcHeight
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << " spitchb: " << uiSrcPitchBytes
                            << std::endl;
#endif
                    }
                };
                //#############################################################################
                //! The CUDA 3D memory copy trait specialization.
                //#############################################################################
                template<>
                struct MemCopyCuda<
                    alpaka::dim::Dim3>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCuda(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind,
                        std::int32_t const & iDev)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        cudaMemcpy3DParms const l_cudaMemcpy3DParms(
                            buildCudaMemcpy3DParms(
                                bufDst,
                                bufSrc,
                                extents,
                                p_cudaMemcpyKind));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            iDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3D(
                                &l_cudaMemcpy3DParms));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCuda(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind,
                        std::int32_t const & iDev,
                        cuda::detail::StreamCuda const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        cudaMemcpy3DParms const l_cudaMemcpy3DParms(
                            buildCudaMemcpy3DParms(
                                bufDst,
                                bufSrc,
                                extents,
                                p_cudaMemcpyKind));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            iDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DAsync(
                                &l_cudaMemcpy3DParms,
                                *stream.m_spCudaStream.get()));
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto buildCudaMemcpy3DParms(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cudaMemcpyKind const & p_cudaMemcpyKind)
                    -> cudaMemcpy3DParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiExtentHeight(extent::getHeight<UInt>(extents));
                        auto const uiExtentDepth(extent::getDepth<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiDstHeight(extent::getHeight<UInt>(bufDst));
                        auto const uiDstDepth(extent::getDepth<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        auto const uiSrcHeight(extent::getHeight<UInt>(bufSrc));
                        auto const uiSrcDepth(extent::getDepth<UInt>(bufSrc));
                        auto const uiDstPitchBytes(mem::getPitchBytes<dim::DimT<TBufDst>::value - 1u, UInt>(bufDst));
                        auto const uiSrcPitchBytes(mem::getPitchBytes<dim::DimT<TBufSrc>::value - 1u, UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentDepth <= uiDstDepth);
                        assert(uiExtentWidth <= uiSrcWidth);
                        assert(uiExtentHeight <= uiSrcHeight);
                        assert(uiExtentDepth <= uiSrcDepth);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DParms l_cudaMemcpy3DParms = {0};
                        //l_cudaMemcpy3DParms.srcArray;     // Either srcArray or srcPtr.
                        //l_cudaMemcpy3DParms.srcPos;       // Optional. Offset in bytes.
                        l_cudaMemcpy3DParms.srcPtr =
                            make_cudaPitchedPtr(
                                reinterpret_cast<void *>(mem::getPtrNative(bufSrc)),
                                uiSrcPitchBytes,
                                uiSrcWidth,
                                uiSrcHeight);
                        //l_cudaMemcpy3DParms.dstArray;     // Either dstArray or dstPtr.
                        //l_cudaMemcpy3DParms.dstPos;       // Optional. Offset in bytes.
                        l_cudaMemcpy3DParms.dstPtr =
                            make_cudaPitchedPtr(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight);
                        l_cudaMemcpy3DParms.extent =
                            make_cudaExtent(
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                uiExtentHeight,
                                uiExtentDepth);
                        l_cudaMemcpy3DParms.kind = p_cudaMemcpyKind;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " eh: " << uiExtentHeight
                            << " ed: " << uiExtentDepth
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dh: " << uiDstHeight
                            << " dd: " << uiDstDepth
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " dpitchb: " << uiDstPitchBytes
                            << " sw: " << uiSrcWidth
                            << " sh: " << uiSrcHeight
                            << " sd: " << uiSrcDepth
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << " spitchb: " << uiSrcPitchBytes
                            << std::endl;
#endif
                        return l_cudaMemcpy3DParms;
                    }
                };

                //#############################################################################
                //! The CUDA peer memory copy trait.
                //#############################################################################
                template<
                    typename TDim>
                struct MemCopyCudaPeer;
                //#############################################################################
                //! The CUDA 1D peer memory copy trait specialization.
                //#############################################################################
                template<>
                struct MemCopyCudaPeer<
                    dim::Dim1>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCudaPeer(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentWidth <= uiSrcWidth);

                        auto const uiDstDev(alpaka::dev::getDev(bufDst).m_iDevice);
                        auto const uiSrcDev(alpaka::dev::getDev(bufSrc).m_iDevice);

                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeer(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                uiDstDev,
                                reinterpret_cast<void const *>(mem::getPtrNative(bufSrc)),
                                uiSrcDev,
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>)));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " ddev: " << uiDstDev
                            << " sw: " << uiSrcWidth
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << " sdev: " << uiSrcDev
                            << std::endl;
#endif
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCudaPeer(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cuda::detail::StreamCuda const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentWidth <= uiSrcWidth);

                        auto const uiDstDev(alpaka::dev::getDev(bufDst).m_iDevice);
                        auto const uiSrcDev(alpaka::dev::getDev(bufSrc).m_iDevice);

                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeerAsync(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                uiDstDev,
                                reinterpret_cast<void const *>(mem::getPtrNative(bufSrc)),
                                uiSrcDev,
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                *stream.m_spCudaStream.get()));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " ddev: " << uiDstDev
                            << " sw: " << uiSrcWidth
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << " sdev: " << uiSrcDev
                            << std::endl;
#endif
                    }
                };
                //#############################################################################
                //! The CUDA 3D peer memory copy trait specialization.
                //#############################################################################
                template<>
                struct MemCopyCudaPeer<
                    dim::Dim3>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCudaPeer(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        cudaMemcpy3DPeerParms const l_cudaMemcpy3DPeerParms(
                            buildCudaMemcpy3DPeerParms(
                                bufDst,
                                bufSrc,
                                extents));

                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeer(
                                &l_cudaMemcpy3DPeerParms));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCudaPeer(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cuda::detail::StreamCuda const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        cudaMemcpy3DPeerParms const l_cudaMemcpy3DPeerParms(
                            buildCudaMemcpy3DPeerParms(
                                bufDst,
                                bufSrc,
                                extents));

                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &l_cudaMemcpy3DPeerParms,
                                *stream.m_spCudaStream.get()));
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto buildCudaMemcpy3DPeerParms(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::ElemT<TBufDst>, typename std::remove_const<mem::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        auto const uiExtentWidth(extent::getWidth<UInt>(extents));
                        auto const uiExtentHeight(extent::getHeight<UInt>(extents));
                        auto const uiExtentDepth(extent::getDepth<UInt>(extents));
                        auto const uiDstWidth(extent::getWidth<UInt>(bufDst));
                        auto const uiDstHeight(extent::getHeight<UInt>(bufDst));
                        auto const uiDstDepth(extent::getDepth<UInt>(bufDst));
                        auto const uiSrcWidth(extent::getWidth<UInt>(bufSrc));
                        auto const uiSrcHeight(extent::getHeight<UInt>(bufSrc));
                        auto const uiSrcDepth(extent::getDepth<UInt>(bufSrc));
                        auto const uiDstPitchBytes(mem::getPitchBytes<dim::DimT<TBufDst>::value - 1u, UInt>(bufDst));
                        auto const uiSrcPitchBytes(mem::getPitchBytes<dim::DimT<TBufSrc>::value - 1u, UInt>(bufSrc));
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentDepth <= uiDstDepth);
                        assert(uiExtentWidth <= uiSrcWidth);
                        assert(uiExtentHeight <= uiSrcHeight);
                        assert(uiExtentDepth <= uiSrcDepth);

                        auto const uiDstDev(alpaka::dev::getDev(bufDst).m_iDevice);
                        auto const uiSrcDev(alpaka::dev::getDev(bufSrc).m_iDevice);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms l_cudaMemcpy3DPeerParms = {0};
                        //l_cudaMemcpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        l_cudaMemcpy3DPeerParms.dstDevice = uiDstDev;
                        //l_cudaMemcpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        l_cudaMemcpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                reinterpret_cast<void *>(mem::getPtrNative(bufDst)),
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight);
                        l_cudaMemcpy3DPeerParms.extent =
                            make_cudaExtent(
                                uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                                uiExtentHeight,
                                uiExtentDepth);
                        //l_cudaMemcpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        l_cudaMemcpy3DPeerParms.srcDevice = uiSrcDev;
                        //l_cudaMemcpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        l_cudaMemcpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                reinterpret_cast<void *>(mem::getPtrNative(bufSrc)),
                                uiSrcPitchBytes,
                                uiSrcWidth,
                                uiSrcHeight);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiExtentWidth
                            << " eh: " << uiExtentHeight
                            << " ed: " << uiExtentDepth
                            << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                            << " dw: " << uiDstWidth
                            << " dh: " << uiDstHeight
                            << " dd: " << uiDstDepth
                            << " dptr: " << reinterpret_cast<void *>(mem::getPtrNative(bufDst))
                            << " dpitchb: " << uiDstPitchBytes
                            << " ddev: " << uiDstDev
                            << " sw: " << uiSrcWidth
                            << " sh: " << uiSrcHeight
                            << " sd: " << uiSrcDepth
                            << " sptr: " << reinterpret_cast<void const *>(mem::getPtrNative(bufSrc))
                            << " spitchb: " << uiSrcPitchBytes
                            << " sdev: " << uiSrcDev
                            << std::endl;
#endif
                        return l_cudaMemcpy3DPeerParms;
                    }
                };
                //#############################################################################
                //! The CUDA 2D peer memory copy trait specialization.
                //#############################################################################
                template<>
                struct MemCopyCudaPeer<
                    dim::Dim2>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCudaPeer(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        // Initiate the memory copy.
                        // NOTE: There is no cudaMemcpy2DPeer so we reuse cudaMemcpy3DPeer.
                        MemCopyCudaPeer<
                            dim::Dim3>
                        ::memCopyCudaPeer(
                            bufDst,
                            bufSrc,
                            extents);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FCT_HOST static auto memCopyCudaPeer(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        cuda::detail::StreamCuda const & stream)
                    -> void
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        // Initiate the memory copy.
                        // NOTE: There is no cudaMemcpy2DPeerAsync so we reuse cudaMemcpy3DPeerAsync.
                        MemCopyCudaPeer<
                            dim::Dim3>
                        ::memCopyCudaPeer(
                            bufDst,
                            bufSrc,
                            extents,
                            stream);
                    }
                };
            }
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for traits::mem::Copy.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The CUDA to Host memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct Copy<
                TDim,
                devs::cpu::detail::DevCpu,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TBufSrc,
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst,
                    TBufSrc const & bufSrc,
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    accs::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyDeviceToHost,
                        alpaka::dev::getDev(bufSrc).m_iDevice);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TBufSrc,
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst,
                    TBufSrc const & bufSrc,
                    TExtents const & extents,
                    accs::cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    accs::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyDeviceToHost,
                        alpaka::dev::getDev(bufSrc).m_iDevice,
                        stream);
                }
            };
            //#############################################################################
            //! The Host to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct Copy<
                TDim,
                accs::cuda::detail::DevCuda,
                devs::cpu::detail::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TBufSrc,
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst,
                    TBufSrc const & bufSrc,
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    accs::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyHostToDevice,
                        alpaka::dev::getDev(bufDst).m_iDevice);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TBufSrc,
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst,
                    TBufSrc const & bufSrc,
                    TExtents const & extents,
                    accs::cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    accs::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyHostToDevice,
                        alpaka::dev::getDev(bufDst).m_iDevice,
                        stream);
                }
            };
            //#############################################################################
            //! The CUDA to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct Copy<
                TDim,
                accs::cuda::detail::DevCuda,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TBufSrc,
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst,
                    TBufSrc const & bufSrc,
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    if(alpaka::dev::getDev(bufDst) == alpaka::dev::getDev(bufSrc))
                    {
                        accs::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                            bufDst,
                            bufSrc,
                            extents,
                            cudaMemcpyDeviceToDevice,
                            alpaka::dev::getDev(bufDst).m_iDevice);
                    }
                    else
                    {
                        accs::cuda::detail::MemCopyCudaPeer<TDim>::memCopyCudaPeer(
                            bufDst,
                            bufSrc,
                            extents);
                    }
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TBufSrc,
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst,
                    TBufSrc const & bufSrc,
                    TExtents const & extents,
                    accs::cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    if(alpaka::dev::getDev(bufDst) == alpaka::dev::getDev(bufSrc))
                    {
                        accs::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                            bufDst,
                            bufSrc,
                            extents,
                            cudaMemcpyDeviceToDevice,
                            alpaka::dev::getDev(bufDst).m_iDevice,
                            stream);
                    }
                    else
                    {
                        accs::cuda::detail::MemCopyCudaPeer<TDim>::memCopyCudaPeer(
                            bufDst,
                            bufSrc,
                            extents,
                            stream);
                    }
                }
            };
        }
    }
}
