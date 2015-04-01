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

#include <alpaka/cuda/mem/Space.hpp>        // SpaceCuda
#include <alpaka/cuda/Stream.hpp>           // StreamCuda
#include <alpaka/cuda/Common.hpp>

#include <alpaka/host/mem/Space.hpp>        // SpaceHost

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>

#include <alpaka/traits/Mem.hpp>            // traits::Copy
#include <alpaka/traits/Extent.hpp>         // traits::getXXX

#include <cassert>                          // assert

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! Page-locks the memory range specified.
            //-----------------------------------------------------------------------------
            /*template<
                typename TBuf>
            auto pageLockHostMem(
                TBuf const & buf)
            -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // cudaHostRegisterDefault: 
                //  See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                // cudaHostRegisterPortable: 
                //  The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
                // cudaHostRegisterMapped: 
                //  Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                //  This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                ALPAKA_CUDA_CHECK_MSG_EXCP_IGNORE(
                    cudaHostRegister(
                        const_cast<void *>(reinterpret_cast<void const *>(mem::getNativePtr(buf))),
                        extent::getProductOfExtents(buf) * sizeof(mem::ElemT<TBuf>),
                        cudaHostRegisterDefault),
                    cudaErrorHostMemoryAlreadyRegistered);
            }
            //-----------------------------------------------------------------------------
            //! Unmaps page-locked memory.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf>
            auto unPageLockHostMem(
                TBuf const & buf)
            -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                ALPAKA_CUDA_CHECK_MSG_EXCP_IGNORE(
                    cudaHostUnregister(
                        const_cast<void *>(reinterpret_cast<void const *>(mem::getNativePtr(buf)))),
                    cudaErrorHostMemoryNotRegistered);
            }*/

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
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::ElemT<TBufDst>, mem::ElemT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiDstWidth(extent::getWidth(bufDst));
                    auto const uiSrcWidth(extent::getWidth(bufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentWidth <= uiSrcWidth);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy(
                            reinterpret_cast<void *>(mem::getNativePtr(bufDst)),
                            reinterpret_cast<void const *>(mem::getNativePtr(bufSrc)),
                            uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                            p_cudaMemcpyKind));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " ew: " << uiExtentWidth
                        << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                        << " dw: " << uiDstWidth
                        << " dptr: " << reinterpret_cast<void *>(mem::getNativePtr(bufDst))
                        << " sw: " << uiSrcWidth
                        << " sptr: " << reinterpret_cast<void const *>(mem::getNativePtr(bufSrc))
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
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::ElemT<TBufDst>, mem::ElemT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiDstWidth(extent::getWidth(bufDst));
                    auto const uiSrcWidth(extent::getWidth(bufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentWidth <= uiSrcWidth);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpyAsync(
                            reinterpret_cast<void *>(mem::getNativePtr(bufDst)),
                            reinterpret_cast<void const *>(mem::getNativePtr(bufSrc)),
                            uiExtentWidth * sizeof(mem::ElemT<TBufDst>),
                            p_cudaMemcpyKind,
                            *stream.m_spCudaStream.get()));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " ew: " << uiExtentWidth
                        << " ewb: " << uiExtentWidth * sizeof(mem::ElemT<TBufDst>)
                        << " dw: " << uiDstWidth
                        << " dptr: " << reinterpret_cast<void *>(mem::getNativePtr(bufDst))
                        << " sw: " << uiSrcWidth
                        << " sptr: " << reinterpret_cast<void const *>(mem::getNativePtr(bufSrc))
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
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::ElemT<TBufDst>, mem::ElemT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiDstWidth(extent::getWidth(bufDst));
                    auto const uiDstHeight(extent::getHeight(bufDst));
                    auto const uiSrcWidth(extent::getWidth(bufSrc));
                    auto const uiSrcHeight(extent::getHeight(bufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy2D(
                            reinterpret_cast<void *>(mem::getNativePtr(bufDst)),
                            mem::getPitchBytes(bufDst),
                            reinterpret_cast<void const *>(mem::getNativePtr(bufSrc)),
                            mem::getPitchBytes(bufSrc),
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
                        << " dptr: " << reinterpret_cast<void *>(mem::getNativePtr(bufDst))
                        << " dpitchb: " << mem::getPitchBytes(bufDst)
                        << " sw: " << uiSrcWidth
                        << " sh: " << uiSrcHeight
                        << " sptr: " << reinterpret_cast<void const *>(mem::getNativePtr(bufSrc))
                        << " spitchb: " << mem::getPitchBytes(bufSrc)
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
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::ElemT<TBufDst>, mem::ElemT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiDstWidth(extent::getWidth(bufDst));
                    auto const uiDstHeight(extent::getHeight(bufDst));
                    auto const uiSrcWidth(extent::getWidth(bufSrc));
                    auto const uiSrcHeight(extent::getHeight(bufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy2DAsync(
                            reinterpret_cast<void *>(mem::getNativePtr(bufDst)),
                            mem::getPitchBytes(bufDst),
                            reinterpret_cast<void const *>(mem::getNativePtr(bufSrc)),
                            mem::getPitchBytes(bufSrc),
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
                        << " dptr: " << reinterpret_cast<void *>(mem::getNativePtr(bufDst))
                        << " dpitchb: " << mem::getPitchBytes(bufDst)
                        << " sw: " << uiSrcWidth
                        << " sh: " << uiSrcHeight
                        << " sptr: " << reinterpret_cast<void const *>(mem::getNativePtr(bufSrc))
                        << " spitchb: " << mem::getPitchBytes(bufSrc)
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
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    cudaMemcpy3DParms const l_cudaMemcpy3DParms(
                        buildCudaMemcpy3DParms(
                            bufDst,
                            bufSrc,
                            extents,
                            p_cudaMemcpyKind));

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
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

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
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
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::DimT<TBufDst>, dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::ElemT<TBufDst>, mem::ElemT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiExtentDepth(extent::getDepth(extents));
                    auto const uiDstWidth(extent::getWidth(bufDst));
                    auto const uiDstHeight(extent::getHeight(bufDst));
                    auto const uiDstDepth(extent::getDepth(bufDst));
                    auto const uiSrcWidth(extent::getWidth(bufSrc));
                    auto const uiSrcHeight(extent::getHeight(bufSrc));
                    auto const uiSrcDepth(extent::getDepth(bufSrc));
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
                            reinterpret_cast<void *>(mem::getNativePtr(bufSrc)),
                            mem::getPitchBytes(bufSrc),
                            uiSrcWidth,
                            uiSrcHeight);
                    //l_cudaMemcpy3DParms.dstArray;     // Either dstArray or dstPtr.
                    //l_cudaMemcpy3DParms.dstPos;       // Optional. Offset in bytes.
                    l_cudaMemcpy3DParms.dstPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(mem::getNativePtr(bufDst)),
                            mem::getPitchBytes(bufDst),
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
                        << " dptr: " << reinterpret_cast<void *>(mem::getNativePtr(bufDst))
                        << " dpitchb: " << mem::getPitchBytes(bufDst)
                        << " sw: " << uiSrcWidth
                        << " sh: " << uiSrcHeight
                        << " sd: " << uiSrcDepth
                        << " sptr: " << reinterpret_cast<void const *>(mem::getNativePtr(bufSrc))
                        << " spitchb: " << mem::getPitchBytes(bufSrc)
                        << std::endl;
#endif
                    return l_cudaMemcpy3DParms;
                }
            };
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
                alpaka::mem::SpaceHost,
                alpaka::mem::SpaceCuda>
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

                    // \TODO: Is memory pinning really useful for synchronous copies?
                    //cuda::detail::pageLockHostMem(bufDst);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyDeviceToHost);

                    //cuda::detail::unPageLockHostMem(bufDst);
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
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    //cuda::detail::pageLockHostMem(bufDst);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyDeviceToHost,
                        stream);

                    //cuda::detail::unPageLockHostMem(bufDst);
                }
            };
            //#############################################################################
            //! The Host to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct Copy<
                TDim,
                alpaka::mem::SpaceCuda,
                alpaka::mem::SpaceHost>
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

                    // \TODO: Is memory pinning really useful for synchronous copies?
                    //cuda::detail::pageLockHostMem(bufSrc);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyHostToDevice);

                    //cuda::detail::unPageLockHostMem(bufSrc);
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
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    //cuda::detail::pageLockHostMem(bufSrc);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyHostToDevice,
                        stream);

                    //cuda::detail::unPageLockHostMem(bufSrc);
                }
            };
            //#############################################################################
            //! The CUDA to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct Copy<
                TDim,
                alpaka::mem::SpaceCuda,
                alpaka::mem::SpaceCuda>
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

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyDeviceToDevice);
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
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        bufDst,
                        bufSrc,
                        extents,
                        cudaMemcpyDeviceToDevice,
                        stream);
                }
            };
        }
    }
}
