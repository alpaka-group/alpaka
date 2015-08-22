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

#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu
#include <alpaka/dev/DevCudaRt.hpp>             // dev::DevCudaRt
#include <alpaka/dim/DimIntegralConst.hpp>      // dim::DimInt<N>
#include <alpaka/extent/Traits.hpp>             // view::getXXX
#include <alpaka/mem/view/Traits.hpp>           // view::Copy
#include <alpaka/stream/StreamCudaRtAsync.hpp>  // stream::StreamCudaRtAsync
#include <alpaka/stream/StreamCudaRtSync.hpp>   // stream::StreamCudaRtSync

#include <alpaka/core/Cuda.hpp>                 // cudaMemcpy, ...

#include <cassert>                              // assert

namespace alpaka
{
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
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    struct TaskCopy;

                    //#############################################################################
                    //! The 1D CUDA memory copy trait.
                    //#############################################################################
                    template<
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    struct TaskCopy<
                        dim::DimInt<1>,
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TBufDst and TBufSrc to have greater or equal range than TExtents.
                        static_assert(
                            std::is_same<mem::view::Elem<TBufDst>, typename std::remove_const<mem::view::Elem<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        using Size = size::Size<TExtents>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopy(
                            TBufDst & bufDst,
                            TBufSrc const & bufSrc,
                            TExtents const & extents,
                            cudaMemcpyKind const & cudaMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extents)),
                                m_dstWidth(static_cast<Size>(extent::getWidth(bufDst))),
                                m_srcWidth(static_cast<Size>(extent::getWidth(bufSrc))),
#endif
                                m_extentWidthBytes(static_cast<Size>(extent::getWidth(extents) * sizeof(mem::view::Elem<TBufDst>))),
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(bufDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentWidth <= m_srcWidth);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ddev: " << m_iDstDevice
                                << " ew: " << m_extentWidth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dptr: " << m_dstMemNative
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sptr: " << m_srcMemNative
                                << std::endl;
                        }
#endif
                        cudaMemcpyKind m_cudaMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_extentWidth;
                        Size m_dstWidth;
                        Size m_srcWidth;
#endif
                        Size m_extentWidthBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 2D CUDA memory copy trait.
                    //#############################################################################
                    template<
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    struct TaskCopy<
                        dim::DimInt<2>,
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TBufDst and TBufSrc to have greater or equal range than TExtents.
                        static_assert(
                            std::is_same<mem::view::Elem<TBufDst>, typename std::remove_const<mem::view::Elem<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        using Size = size::Size<TExtents>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopy(
                            TBufDst & bufDst,
                            TBufSrc const & bufSrc,
                            TExtents const & extents,
                            cudaMemcpyKind const & cudaMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extents)),
#endif
                                m_extentWidthBytes(static_cast<Size>(extent::getWidth(extents) * sizeof(mem::view::Elem<TBufDst>))),
                                m_dstWidth(static_cast<Size>(extent::getWidth(bufDst))),      // required for 3D peer copy
                                m_srcWidth(static_cast<Size>(extent::getWidth(bufSrc))),      // required for 3D peer copy

                                m_extentHeight(extent::getHeight(extents)),
                                m_dstHeight(static_cast<Size>(extent::getHeight(bufDst))),    // required for 3D peer copy
                                m_srcHeight(static_cast<Size>(extent::getHeight(bufSrc))),    // required for 3D peer copy

                                m_dstPitchBytes(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TBufDst>::value - 1u>(bufDst))),
                                m_srcPitchBytes(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TBufSrc>::value - 1u>(bufSrc))),

                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(bufDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentWidthBytes <= m_dstPitchBytes);
                            assert(m_extentWidthBytes <= m_srcPitchBytes);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstPitchBytes
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcPitchBytes
                                << std::endl;
                        }
#endif
                        cudaMemcpyKind m_cudaMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_extentWidth;
#endif
                        Size m_extentWidthBytes;
                        Size m_dstWidth;          // required for 3D peer copy
                        Size m_srcWidth;          // required for 3D peer copy

                        Size m_extentHeight;
                        Size m_dstHeight;         // required for 3D peer copy
                        Size m_srcHeight;         // required for 3D peer copy

                        Size m_dstPitchBytes;
                        Size m_srcPitchBytes;

                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 3D CUDA memory copy trait.
                    //#############################################################################
                    template<
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    struct TaskCopy<
                        dim::DimInt<3>,
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TBufDst and TBufSrc to have greater or equal range than TExtents.
                        static_assert(
                            std::is_same<mem::view::Elem<TBufDst>, typename std::remove_const<mem::view::Elem<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        using Size = size::Size<TExtents>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopy(
                            TBufDst & bufDst,
                            TBufSrc const & bufSrc,
                            TExtents const & extents,
                            cudaMemcpyKind const & cudaMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemCpyKind),

                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),

                                m_extentWidth(extent::getWidth(extents)),
                                m_extentWidthBytes(static_cast<Size>(m_extentWidth * sizeof(mem::view::Elem<TBufDst>))),
                                m_dstWidth(static_cast<Size>(extent::getWidth(bufDst))),
                                m_srcWidth(static_cast<Size>(extent::getWidth(bufSrc))),

                                m_extentHeight(extent::getHeight(extents)),
                                m_dstHeight(static_cast<Size>(extent::getHeight(bufDst))),
                                m_srcHeight(static_cast<Size>(extent::getHeight(bufSrc))),

                                m_extentDepth(extent::getDepth(extents)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Size>(extent::getDepth(bufDst))),
                                m_srcDepth(static_cast<Size>(extent::getDepth(bufSrc))),
#endif
                                m_dstPitchBytes(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TBufDst>::value - 1u>(bufDst))),
                                m_srcPitchBytes(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TBufSrc>::value - 1u>(bufSrc))),

                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(bufDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentDepth <= m_dstDepth);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentDepth <= m_srcDepth);
                            assert(m_extentWidthBytes <= m_dstPitchBytes);
                            assert(m_extentWidthBytes <= m_srcPitchBytes);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ed: " << m_extentDepth
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dd: " << m_dstDepth
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstPitchBytes
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sd: " << m_srcDepth
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcPitchBytes
                                << std::endl;
                        }
#endif
                        cudaMemcpyKind m_cudaMemCpyKind;

                        int m_iDstDevice;
                        int m_iSrcDevice;

                        Size m_extentWidth;
                        Size m_extentWidthBytes;
                        Size m_dstWidth;
                        Size m_srcWidth;

                        Size m_extentHeight;
                        Size m_dstHeight;
                        Size m_srcHeight;

                        Size m_extentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_dstDepth;
                        Size m_srcDepth;
#endif
                        Size m_dstPitchBytes;
                        Size m_srcPitchBytes;

                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for view::TaskCopy.
            //-----------------------------------------------------------------------------
            namespace traits
            {
                //#############################################################################
                //! The CUDA to CPU memory copy trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> mem::view::cuda::detail::TaskCopy<
                        TDim,
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(bufSrc).m_iDevice);

                        return
                            mem::view::cuda::detail::TaskCopy<
                                TDim,
                                TBufDst,
                                TBufSrc,
                                TExtents>(
                                    bufDst,
                                    bufSrc,
                                    extents,
                                    cudaMemcpyDeviceToHost,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CPU to CUDA memory copy trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCudaRt,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> mem::view::cuda::detail::TaskCopy<
                        TDim,
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(bufDst).m_iDevice);

                        return
                            mem::view::cuda::detail::TaskCopy<
                                TDim,
                                TBufDst,
                                TBufSrc,
                                TExtents>(
                                    bufDst,
                                    bufSrc,
                                    extents,
                                    cudaMemcpyHostToDevice,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CUDA to CUDA memory copy trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCudaRt,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> mem::view::cuda::detail::TaskCopy<
                        TDim,
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::cuda::detail::TaskCopy<
                                TDim,
                                TBufDst,
                                TBufSrc,
                                TExtents>(
                                    bufDst,
                                    bufSrc,
                                    extents,
                                    cudaMemcpyDeviceToDevice,
                                    dev::getDev(bufDst).m_iDevice,
                                    dev::getDev(bufSrc).m_iDevice);
                    }
                };
            }
            namespace cuda
            {
                namespace detail
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FN_HOST static auto buildCudaMemcpy3DParms(
                        mem::view::cuda::detail::TaskCopy<dim::DimInt<3>, TBufDst, TBufSrc, TExtents> const & task)
                    -> cudaMemcpy3DParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        auto const & dstHeight(task.m_dstHeight);
                        auto const & srcHeight(task.m_srcHeight);

                        auto const & extentDepth(task.m_extentDepth);

                        auto const & dstPitchBytes(task.m_dstPitchBytes);
                        auto const & srcPitchBytes(task.m_srcPitchBytes);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DParms cudaMemCpy3DParms = {0};
                        //cudaMemCpy3DParms.srcArray;     // Either srcArray or srcPtr.
                        //cudaMemCpy3DParms.srcPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                srcPitchBytes,
                                srcWidth,
                                srcHeight);
                        //cudaMemCpy3DParms.dstArray;     // Either dstArray or dstPtr.
                        //cudaMemCpy3DParms.dstPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                dstPitchBytes,
                                dstWidth,
                                dstHeight);
                        cudaMemCpy3DParms.extent =
                            make_cudaExtent(
                                extentWidthBytes,
                                extentHeight,
                                extentDepth);
                        cudaMemCpy3DParms.kind = task.m_cudaMemCpyKind;

                        return cudaMemCpy3DParms;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    ALPAKA_FN_HOST static auto buildCudaMemcpy3DPeerParms(
                        mem::view::cuda::detail::TaskCopy<dim::DimInt<2>, TBufDst, TBufSrc, TExtents> const & task)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & iDstDev(task.m_iDstDevice);
                        auto const & iSrcDev(task.m_iSrcDevice);

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        auto const & dstHeight(task.m_dstHeight);
                        auto const & srcHeight(task.m_srcHeight);

                        auto const extentDepth(1u);

                        auto const & dstPitchBytes(task.m_dstPitchBytes);
                        auto const & srcPitchBytes(task.m_srcPitchBytes);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms = {0};
                        //cudaMemCpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        //cudaMemCpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                dstPitchBytes,
                                dstWidth,
                                dstHeight);
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                extentWidthBytes,
                                extentHeight,
                                extentDepth);
                        //cudaMemCpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        //cudaMemCpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                srcPitchBytes,
                                srcWidth,
                                srcHeight);

                        return cudaMemCpy3DPeerParms;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    ALPAKA_FN_HOST static auto buildCudaMemcpy3DPeerParms(
                        mem::view::cuda::detail::TaskCopy<dim::DimInt<3>, TBufDst, TBufSrc, TExtents> const & task)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & iDstDev(task.m_iDstDevice);
                        auto const & iSrcDev(task.m_iSrcDevice);

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        auto const & dstHeight(task.m_dstHeight);
                        auto const & srcHeight(task.m_srcHeight);

                        auto const & extentDepth(task.m_extentDepth);

                        auto const & dstPitchBytes(task.m_dstPitchBytes);
                        auto const & srcPitchBytes(task.m_srcPitchBytes);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms = {0};
                        //cudaMemCpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        //cudaMemCpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                dstPitchBytes,
                                dstWidth,
                                dstHeight);
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                extentWidthBytes,
                                extentHeight,
                                extentDepth);
                        //cudaMemCpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        //cudaMemCpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                srcPitchBytes,
                                srcWidth,
                                srcHeight);

                        return cudaMemCpy3DPeerParms;
                    }
                }
            }
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA async device stream 1D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TBufSrc,
                typename TBufDst>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TBufDst, TBufSrc, TExtents>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TBufDst, TBufSrc, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                extentWidthBytes,
                                cudaMemCpyKind,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                extentWidthBytes,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 1D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TBufSrc,
                typename TBufDst>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TBufDst, TBufSrc, TExtents>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TBufDst, TBufSrc, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy(
                                dstNativePtr,
                                srcNativePtr,
                                extentWidthBytes,
                                cudaMemCpyKind));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeer(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                extentWidthBytes));
                    }
                }
            };
            //#############################################################################
            //! The CUDA async device stream 2D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TBufSrc,
                typename TBufDst>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TBufDst, TBufSrc, TExtents>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TBufDst, TBufSrc, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytes(task.m_dstPitchBytes);
                        auto const & srcPitchBytes(task.m_srcPitchBytes);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2DAsync(
                                dstNativePtr,
                                dstPitchBytes,
                                srcNativePtr,
                                srcPitchBytes,
                                extentWidthBytes,
                                extentHeight,
                                cudaMemCpyKind,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                    else
                    {
                        // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 2D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TBufSrc,
                typename TBufDst>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TBufDst, TBufSrc, TExtents>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TBufDst, TBufSrc, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytes(task.m_dstPitchBytes);
                        auto const & srcPitchBytes(task.m_srcPitchBytes);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2D(
                                dstNativePtr,
                                dstPitchBytes,
                                srcNativePtr,
                                srcPitchBytes,
                                extentWidthBytes,
                                extentHeight,
                                cudaMemCpyKind));
                    }
                    else
                    {
                        // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeer(
                                &cudaMemCpy3DPeerParms));
                    }
                }
            };
            //#############################################################################
            //! The CUDA async device stream 3D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TBufSrc,
                typename TBufDst>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TBufDst, TBufSrc, TExtents>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TBufDst, TBufSrc, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DParms const cudaMemCpy3DParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DParms(
                                task));
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DAsync(
                                &cudaMemCpy3DParms,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                    else
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 3D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TBufSrc,
                typename TBufDst>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TBufDst, TBufSrc, TExtents>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TBufDst, TBufSrc, TExtents> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DParms const cudaMemCpy3DParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DParms(
                                task));
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3D(
                                &cudaMemCpy3DParms));
                    }
                    else
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeer(
                                &cudaMemCpy3DPeerParms));
                    }
                }
            };
        }
    }
}
