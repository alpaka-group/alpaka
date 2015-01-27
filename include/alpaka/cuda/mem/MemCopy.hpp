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

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>

#include <alpaka/traits/Mem.hpp>            // traits::MemCopy
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <alpaka/host/MemSpace.hpp>         // MemSpaceHost

#include <alpaka/cuda/MemSpace.hpp>         // MemSpaceCuda
#include <alpaka/cuda/Stream.hpp>           // StreamCuda
#include <alpaka/cuda/Common.hpp>

#include <cstddef>                          // std::size_t
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
            template<
                typename TMemBuf>
            void pageLockHostMem(
                TMemBuf const & memBuf)
            {
                // cudaHostRegisterDefault: 
                //  See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                // cudaHostRegisterPortable: 
                //  The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
                // cudaHostRegisterMapped: 
                //  Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                //  This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                ALPAKA_CUDA_CHECK(
                    cudaHostRegister(
                        const_cast<void *>(reinterpret_cast<void const *>(mem::getNativePtr(memBuf))),
                        extent::getProductOfExtents(memBuf) * sizeof(mem::GetMemElemT<TMemBuf>),
                        cudaHostRegisterDefault));
            }
            //-----------------------------------------------------------------------------
            //! Unmaps page-locked memory.
            //-----------------------------------------------------------------------------
            template<
                typename TMemBuf>
            void unPageLockHostMem(
                TMemBuf const & memBuf)
            {
                ALPAKA_CUDA_CHECK(
                    cudaHostUnregister(
                        const_cast<void *>(reinterpret_cast<void const *>(mem::getNativePtr(memBuf)))));
            }

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
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                {
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::GetMemElemT<TMemBufDst>, mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiDstWidth(extent::getWidth(memBufDst));
                    auto const uiSrcWidth(extent::getWidth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentWidth <= uiSrcWidth);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy(
                            reinterpret_cast<void *>(mem::getNativePtr(memBufDst)),
                            reinterpret_cast<void const *>(mem::getNativePtr(memBufSrc)),
                            uiExtentWidth * sizeof(mem::GetMemElemT<TMemBufDst>),
                            p_cudaMemcpyKind));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind,
                    cuda::detail::StreamCuda const & stream)
                {
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::GetMemElemT<TMemBufDst>, mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiDstWidth(extent::getWidth(memBufDst));
                    auto const uiSrcWidth(extent::getWidth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentWidth <= uiSrcWidth);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpyAsync(
                            reinterpret_cast<void *>(mem::getNativePtr(memBufDst)),
                            reinterpret_cast<void const *>(mem::getNativePtr(memBufSrc)),
                            uiExtentWidth * sizeof(mem::GetMemElemT<TMemBufDst>),
                            p_cudaMemcpyKind,
                            stream.m_cudaStream));
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
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                {
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::GetMemElemT<TMemBufDst>, mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiDstWidth(extent::getWidth(memBufDst));
                    auto const uiDstHeight(extent::getHeight(memBufDst));
                    auto const uiSrcWidth(extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(extent::getHeight(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy2D(
                            reinterpret_cast<void *>(mem::getNativePtr(memBufDst)),
                            mem::getPitchBytes(memBufDst),
                            reinterpret_cast<void const *>(mem::getNativePtr(memBufSrc)),
                            mem::getPitchBytes(memBufSrc),
                            uiExtentWidth * sizeof(mem::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            p_cudaMemcpyKind));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind,
                    cuda::detail::StreamCuda const & stream)
                {
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::GetMemElemT<TMemBufDst>, mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiDstWidth(extent::getWidth(memBufDst));
                    auto const uiDstHeight(extent::getHeight(memBufDst));
                    auto const uiSrcWidth(extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(extent::getHeight(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy2D(
                            reinterpret_cast<void *>(mem::getNativePtr(memBufDst)),
                            mem::getPitchBytes(memBufDst),
                            reinterpret_cast<void const *>(mem::getNativePtr(memBufSrc)),
                            mem::getPitchBytes(memBufSrc),
                            uiExtentWidth * sizeof(mem::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            p_cudaMemcpyKind,
                            stream.m_cudaStream));
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
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                {
                    cudaMemcpy3DParms const l_cudaMemcpy3DParms(
                        buildCudaMemcpy3DParms(
                            memBufDst,
                            memBufSrc,
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
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind,
                    cuda::detail::StreamCuda const & stream)
                {
                    cudaMemcpy3DParms const l_cudaMemcpy3DParms(
                        buildCudaMemcpy3DParms(
                            memBufDst,
                            memBufSrc,
                            extents,
                            p_cudaMemcpyKind));

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy3DAsync(
                            &l_cudaMemcpy3DParms,
                            stream.m_cudaStream));
                }
            private:
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static cudaMemcpy3DParms buildCudaMemcpy3DParms(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & p_cudaMemcpyKind)
                {
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<mem::GetMemElemT<TMemBufDst>, mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(extent::getWidth(extents));
                    auto const uiExtentHeight(extent::getHeight(extents));
                    auto const uiExtentDepth(extent::getDepth(extents));
                    auto const uiDstWidth(extent::getWidth(memBufDst));
                    auto const uiDstHeight(extent::getHeight(memBufDst));
                    auto const uiDstDepth(extent::getDepth(memBufDst));
                    auto const uiSrcWidth(extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(extent::getHeight(memBufSrc));
                    auto const uiSrcDepth(extent::getDepth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);
                    assert(uiExtentDepth <= uiSrcDepth);

                    // Fill CUDA parameter structure.
                    cudaMemcpy3DParms l_cudaMemcpy3DParms = {0};
                    //l_cudaMemcpy3DParms.srcArray;    // Either srcArray or srcPtr.
                    //l_cudaMemcpy3DParms.srcPos;      // Optional. Offset in bytes.
                    l_cudaMemcpy3DParms.srcPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(mem::getNativePtr(memBufSrc)),
                            mem::getPitchBytes(memBufSrc),
                            uiSrcWidth,
                            uiSrcHeight);
                    //l_cudaMemcpy3DParms.dstArray;    // Either dstArray or dstPtr.
                    //l_cudaMemcpy3DParms.dstPos;      // Optional. Offset in bytes.
                    l_cudaMemcpy3DParms.dstPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(mem::getNativePtr(memBufDst)),
                            mem::getPitchBytes(memBufDst),
                            uiDstWidth,
                            uiDstHeight);
                    l_cudaMemcpy3DParms.extent = 
                        make_cudaExtent(
                            uiExtentWidth * sizeof(mem::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            uiExtentDepth);
                    l_cudaMemcpy3DParms.kind = p_cudaMemcpyKind;

                    return l_cudaMemcpy3DParms;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for traits::mem::MemCopy.
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
            struct MemCopy<
                TDim,
                alpaka::mem::MemSpaceHost,
                alpaka::mem::MemSpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst,
                    TMemBufSrc const & memBufSrc,
                    TExtents const & extents)
                {
                    // \TODO: Is memory pinning really useful for synchronous copies?
                    cuda::detail::pageLockHostMem(memBufDst);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyDeviceToHost);

                    cuda::detail::unPageLockHostMem(memBufDst);
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst,
                    TMemBufSrc const & memBufSrc,
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                {
                    cuda::detail::pageLockHostMem(memBufDst);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyDeviceToHost,
                        stream);

                    cuda::detail::unPageLockHostMem(memBufDst);
                }
            };
            //#############################################################################
            //! The Host to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim,
                alpaka::mem::MemSpaceCuda,
                alpaka::mem::MemSpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents)
                {
                    // \TODO: Is memory pinning really useful for synchronous copies?
                    cuda::detail::pageLockHostMem(memBufSrc);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyHostToDevice);

                    cuda::detail::unPageLockHostMem(memBufSrc);
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                {
                    cuda::detail::pageLockHostMem(memBufSrc);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyHostToDevice,
                        stream);

                    cuda::detail::unPageLockHostMem(memBufSrc);
                }
            };
            //#############################################################################
            //! The CUDA to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim,
                alpaka::mem::MemSpaceCuda,
                alpaka::mem::MemSpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents)
                {
                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyDeviceToDevice);
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                {
                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyDeviceToDevice,
                        stream);
                }
            };
        }
    }
}
