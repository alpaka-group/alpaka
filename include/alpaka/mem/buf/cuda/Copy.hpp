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
                                m_uiExtentWidth(extent::getWidth(extents)),
                                m_uiDstWidth(extent::getWidth(bufDst)),
                                m_uiSrcWidth(extent::getWidth(bufSrc)),
#endif
                                m_uiExtentWidthBytes(extent::getWidth(extents) * sizeof(mem::view::Elem<TBufDst>)),
                                m_pDstNative(reinterpret_cast<void *>(mem::view::getPtrNative(bufDst))),
                                m_pSrcNative(reinterpret_cast<void const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_uiExtentWidth <= m_uiDstWidth);
                            assert(m_uiExtentWidth <= m_uiSrcWidth);
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
                                << " ew: " << m_uiExtentWidth
                                << " ewb: " << m_uiExtentWidthBytes
                                << " dw: " << m_uiDstWidth
                                << " dptr: " << m_pDstNative
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_uiSrcWidth
                                << " sptr: " << m_pSrcNative
                                << std::endl;
                        }
#endif
                        cudaMemcpyKind m_cudaMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_uiExtentWidth;
                        Size m_uiDstWidth;
                        Size m_uiSrcWidth;
#endif
                        Size m_uiExtentWidthBytes;
                        void * m_pDstNative;
                        void const * m_pSrcNative;
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
                                m_uiExtentWidth(extent::getWidth(extents)),
#endif
                                m_uiExtentWidthBytes(extent::getWidth(extents) * sizeof(mem::view::Elem<TBufDst>)),
                                m_uiDstWidth(extent::getWidth(bufDst)),         // required for 3D peer copy
                                m_uiSrcWidth(extent::getWidth(bufSrc)),         // required for 3D peer copy

                                m_uiExtentHeight(extent::getHeight(extents)),
                                m_uiDstHeight(extent::getHeight(bufDst)),       // required for 3D peer copy
                                m_uiSrcHeight(extent::getHeight(bufSrc)),       // required for 3D peer copy

                                m_uiDstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBufDst>::value - 1u>(bufDst)),
                                m_uiSrcPitchBytes(mem::view::getPitchBytes<dim::Dim<TBufSrc>::value - 1u>(bufSrc)),

                                m_pDstNative(reinterpret_cast<void *>(mem::view::getPtrNative(bufDst))),
                                m_pSrcNative(reinterpret_cast<void const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_uiExtentWidth <= m_uiDstWidth);
                            assert(m_uiExtentHeight <= m_uiDstHeight);
                            assert(m_uiExtentWidth <= m_uiSrcWidth);
                            assert(m_uiExtentHeight <= m_uiSrcHeight);
                            assert(m_uiExtentWidthBytes <= m_uiDstPitchBytes);
                            assert(m_uiExtentWidthBytes <= m_uiSrcPitchBytes);
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
                                << " ew: " << m_uiExtentWidth
                                << " eh: " << m_uiExtentHeight
                                << " ewb: " << m_uiExtentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_uiDstWidth
                                << " dh: " << m_uiDstHeight
                                << " dptr: " << m_pDstNative
                                << " dpitchb: " << m_uiDstPitchBytes
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_uiSrcWidth
                                << " sh: " << m_uiSrcHeight
                                << " sptr: " << m_pSrcNative
                                << " spitchb: " << m_uiSrcPitchBytes
                                << std::endl;
                        }
#endif
                        cudaMemcpyKind m_cudaMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_uiExtentWidth;
#endif
                        Size m_uiExtentWidthBytes;
                        Size m_uiDstWidth;          // required for 3D peer copy
                        Size m_uiSrcWidth;          // required for 3D peer copy

                        Size m_uiExtentHeight;
                        Size m_uiDstHeight;         // required for 3D peer copy
                        Size m_uiSrcHeight;         // required for 3D peer copy

                        Size m_uiDstPitchBytes;
                        Size m_uiSrcPitchBytes;

                        void * m_pDstNative;
                        void const * m_pSrcNative;
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

                                m_uiExtentWidth(extent::getWidth(extents)),
                                m_uiExtentWidthBytes(m_uiExtentWidth * sizeof(mem::view::Elem<TBufDst>)),
                                m_uiDstWidth(extent::getWidth(bufDst)),
                                m_uiSrcWidth(extent::getWidth(bufSrc)),

                                m_uiExtentHeight(extent::getHeight(extents)),
                                m_uiDstHeight(extent::getHeight(bufDst)),
                                m_uiSrcHeight(extent::getHeight(bufSrc)),

                                m_uiExtentDepth(extent::getDepth(extents)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_uiDstDepth(extent::getDepth(bufDst)),
                                m_uiSrcDepth(extent::getDepth(bufSrc)),
#endif
                                m_uiDstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBufDst>::value - 1u>(bufDst)),
                                m_uiSrcPitchBytes(mem::view::getPitchBytes<dim::Dim<TBufSrc>::value - 1u>(bufSrc)),

                                m_pDstNative(reinterpret_cast<void *>(mem::view::getPtrNative(bufDst))),
                                m_pSrcNative(reinterpret_cast<void const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_uiExtentWidth <= m_uiDstWidth);
                            assert(m_uiExtentHeight <= m_uiDstHeight);
                            assert(m_uiExtentDepth <= m_uiDstDepth);
                            assert(m_uiExtentWidth <= m_uiSrcWidth);
                            assert(m_uiExtentHeight <= m_uiSrcHeight);
                            assert(m_uiExtentDepth <= m_uiSrcDepth);
                            assert(m_uiExtentWidthBytes <= m_uiDstPitchBytes);
                            assert(m_uiExtentWidthBytes <= m_uiSrcPitchBytes);
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
                                << " ew: " << m_uiExtentWidth
                                << " eh: " << m_uiExtentHeight
                                << " ed: " << m_uiExtentDepth
                                << " ewb: " << m_uiExtentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_uiDstWidth
                                << " dh: " << m_uiDstHeight
                                << " dd: " << m_uiDstDepth
                                << " dptr: " << m_pDstNative
                                << " dpitchb: " << m_uiDstPitchBytes
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_uiSrcWidth
                                << " sh: " << m_uiSrcHeight
                                << " sd: " << m_uiSrcDepth
                                << " sptr: " << m_pSrcNative
                                << " spitchb: " << m_uiSrcPitchBytes
                                << std::endl;
                        }
#endif
                        cudaMemcpyKind m_cudaMemCpyKind;

                        int m_iDstDevice;
                        int m_iSrcDevice;

                        Size m_uiExtentWidth;
                        Size m_uiExtentWidthBytes;
                        Size m_uiDstWidth;
                        Size m_uiSrcWidth;

                        Size m_uiExtentHeight;
                        Size m_uiDstHeight;
                        Size m_uiSrcHeight;

                        Size m_uiExtentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_uiDstDepth;
                        Size m_uiSrcDepth;
#endif
                        Size m_uiDstPitchBytes;
                        Size m_uiSrcPitchBytes;

                        void * m_pDstNative;
                        void const * m_pSrcNative;
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

                        auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);
                        auto const & uiDstWidth(task.m_uiDstWidth);
                        auto const & uiSrcWidth(task.m_uiSrcWidth);

                        auto const & uiExtentHeight(task.m_uiExtentHeight);
                        auto const & uiDstHeight(task.m_uiDstHeight);
                        auto const & uiSrcHeight(task.m_uiSrcHeight);

                        auto const & uiExtentDepth(task.m_uiExtentDepth);

                        auto const & uiDstPitchBytes(task.m_uiDstPitchBytes);
                        auto const & uiSrcPitchBytes(task.m_uiSrcPitchBytes);

                        auto const & pDstNativePtr(task.m_pDstNative);
                        auto const & pSrcNativePtr(task.m_pSrcNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DParms cudaMemCpy3DParms = {0};
                        //cudaMemCpy3DParms.srcArray;     // Either srcArray or srcPtr.
                        //cudaMemCpy3DParms.srcPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DParms.srcPtr =
                            make_cudaPitchedPtr(
                                pSrcNativePtr,
                                uiSrcPitchBytes,
                                uiSrcWidth,
                                uiSrcHeight);
                        //cudaMemCpy3DParms.dstArray;     // Either dstArray or dstPtr.
                        //cudaMemCpy3DParms.dstPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DParms.dstPtr =
                            make_cudaPitchedPtr(
                                pDstNativePtr,
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight);
                        cudaMemCpy3DParms.extent =
                            make_cudaExtent(
                                uiExtentWidthBytes,
                                uiExtentHeight,
                                uiExtentDepth);
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

                        auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);
                        auto const & uiDstWidth(task.m_uiDstWidth);
                        auto const & uiSrcWidth(task.m_uiSrcWidth);

                        auto const & uiExtentHeight(task.m_uiExtentHeight);
                        auto const & uiDstHeight(task.m_uiDstHeight);
                        auto const & uiSrcHeight(task.m_uiSrcHeight);

                        auto const uiExtentDepth(1u);

                        auto const & uiDstPitchBytes(task.m_uiDstPitchBytes);
                        auto const & uiSrcPitchBytes(task.m_uiSrcPitchBytes);

                        auto const & pDstNativePtr(task.m_pDstNative);
                        auto const & pSrcNativePtr(task.m_pSrcNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms = {0};
                        //cudaMemCpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        //cudaMemCpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                pDstNativePtr,
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight);
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                uiExtentWidthBytes,
                                uiExtentHeight,
                                uiExtentDepth);
                        //cudaMemCpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        //cudaMemCpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(pSrcNativePtr),
                                uiSrcPitchBytes,
                                uiSrcWidth,
                                uiSrcHeight);

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

                        auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);
                        auto const & uiDstWidth(task.m_uiDstWidth);
                        auto const & uiSrcWidth(task.m_uiSrcWidth);

                        auto const & uiExtentHeight(task.m_uiExtentHeight);
                        auto const & uiDstHeight(task.m_uiDstHeight);
                        auto const & uiSrcHeight(task.m_uiSrcHeight);

                        auto const & uiExtentDepth(task.m_uiExtentDepth);

                        auto const & uiDstPitchBytes(task.m_uiDstPitchBytes);
                        auto const & uiSrcPitchBytes(task.m_uiSrcPitchBytes);

                        auto const & pDstNativePtr(task.m_pDstNative);
                        auto const & pSrcNativePtr(task.m_pSrcNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms = {0};
                        //cudaMemCpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        //cudaMemCpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                pDstNativePtr,
                                uiDstPitchBytes,
                                uiDstWidth,
                                uiDstHeight);
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                uiExtentWidthBytes,
                                uiExtentHeight,
                                uiExtentDepth);
                        //cudaMemCpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        //cudaMemCpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                pSrcNativePtr,
                                uiSrcPitchBytes,
                                uiSrcWidth,
                                uiSrcHeight);

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

                    auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);

                    auto const & pDstNativePtr(task.m_pDstNative);
                    auto const & pSrcNativePtr(task.m_pSrcNative);

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
                                pDstNativePtr,
                                pSrcNativePtr,
                                uiExtentWidthBytes,
                                cudaMemCpyKind,
                                stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeerAsync(
                                pDstNativePtr,
                                iDstDev,
                                pSrcNativePtr,
                                iSrcDev,
                                uiExtentWidthBytes,
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

                    auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);

                    auto const & pDstNativePtr(task.m_pDstNative);
                    auto const & pSrcNativePtr(task.m_pSrcNative);

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
                                pDstNativePtr,
                                pSrcNativePtr,
                                uiExtentWidthBytes,
                                cudaMemCpyKind));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeer(
                                pDstNativePtr,
                                iDstDev,
                                pSrcNativePtr,
                                iSrcDev,
                                uiExtentWidthBytes));
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
                        auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);
                        auto const & uiExtentHeight(task.m_uiExtentHeight);

                        auto const & uiDstPitchBytes(task.m_uiDstPitchBytes);
                        auto const & uiSrcPitchBytes(task.m_uiSrcPitchBytes);

                        auto const & pDstNativePtr(task.m_pDstNative);
                        auto const & pSrcNativePtr(task.m_pSrcNative);

                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2DAsync(
                                pDstNativePtr,
                                uiDstPitchBytes,
                                pSrcNativePtr,
                                uiSrcPitchBytes,
                                uiExtentWidthBytes,
                                uiExtentHeight,
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
                        auto const & uiExtentWidthBytes(task.m_uiExtentWidthBytes);
                        auto const & uiExtentHeight(task.m_uiExtentHeight);

                        auto const & uiDstPitchBytes(task.m_uiDstPitchBytes);
                        auto const & uiSrcPitchBytes(task.m_uiSrcPitchBytes);

                        auto const & pDstNativePtr(task.m_pDstNative);
                        auto const & pSrcNativePtr(task.m_pSrcNative);

                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2D(
                                pDstNativePtr,
                                uiDstPitchBytes,
                                pSrcNativePtr,
                                uiSrcPitchBytes,
                                uiExtentWidthBytes,
                                uiExtentHeight,
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
