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

#include <alpaka/traits/Mem.hpp>            // traits::MemCopy
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/BasicExtents.hpp>     // extent::BasicExtents<TDim>

#include <alpaka/cuda/MemSpace.hpp>         // MemSpaceCuda
#include <alpaka/host/MemSpace.hpp>         // MemSpaceHost

#include <alpaka/cuda/Common.hpp>

#include <cstddef>                          // std::size_t
#include <cassert>                          // assert
#include <memory>                           // std::shared_ptr

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA memory buffer.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            class MemBufCuda :
                public alpaka::extent::BasicExtents<TDim>
            {
            public:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                MemBufCuda(
                    TElem * pMem,
                    std::size_t const & uiPitchBytes,
                    TExtents const & extents) :
                        BasicExtents<TDim>(extents),
                        m_spMem(
                            pMem,
                            [](TElem * pBuffer)
                            {
                                assert(pBuffer);
                                cudaFree(reinterpret_cast<void *>(pBuffer));
                            }),
                        m_uiPitchBytes(uiPitchBytes)
                {
                    static_assert(
                        std::is_same<TDim, alpaka::dim::GetDimT<TExtents>>::value,
                        "The extents are required to have the same dimensionality as the MemBufCuda!");
                }

            public:
                std::shared_ptr<TElem> m_spMem;

                std::size_t m_uiPitchBytes; // \TODO: By using class specialization for Dim1 we could remove this value in this case.
            };

            //#############################################################################
            //! Page-locks the memory range specified.
            //#############################################################################
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
                        const_cast<void *>(reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf))), 
                        alpaka::extent::getProductOfExtents(memBuf) * sizeof(alpaka::mem::GetMemElemT<TMemBuf>),
                        cudaHostRegisterDefault));
            }
            //#############################################################################
            //! Unmaps page-locked memory.
            //#############################################################################
            template<
                typename TMemBuf>
            void unPageLockHostMem(
                TMemBuf const & memBuf)
            {
                ALPAKA_CUDA_CHECK(
                    cudaHostUnregister(
                        const_cast<void *>(reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf)))));
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
                alpaka::dim::Dim1>
            {
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & Extents, 
                    cudaMemcpyKind const & cudaMemcpyKindVal)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::mem::GetMemElemT<TMemBufDst>, alpaka::mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentWidth <= uiSrcWidth);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBufDst)),
                            reinterpret_cast<void const *>(alpaka::mem::getNativePtr(memBufSrc)),
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBufDst>),
                            cudaMemcpyKindVal));
                }
            };
            //#############################################################################
            //! The CUDA 2D memory copy trait specialization.
            //#############################################################################
            template<>
            struct MemCopyCuda<
                alpaka::dim::Dim2>
            {
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & cudaMemcpyKindVal)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::mem::GetMemElemT<TMemBufDst>, alpaka::mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy2D(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBufDst)),
                            alpaka::mem::getPitchBytes(memBufDst),
                            reinterpret_cast<void const *>(alpaka::mem::getNativePtr(memBufSrc)),
                            alpaka::mem::getPitchBytes(memBufSrc),
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            cudaMemcpyKindVal));
                }
            };
            //#############################################################################
            //! The CUDA 3D memory copy trait specialization.
            //#############################################################################
            template<>
            struct MemCopyCuda<
                alpaka::dim::Dim3>
            {
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents, 
                    cudaMemcpyKind const & cudaMemcpyKindVal)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::mem::GetMemElemT<TMemBufDst>, alpaka::mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBufDst));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(memBufSrc));
                    auto const uiSrcDepth(alpaka::extent::getDepth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);
                    assert(uiExtentDepth <= uiSrcDepth);

                    // Fill CUDA parameter structure.
                    cudaMemcpy3DParms cudaMemcpy3DParmsVal = {0};
                    //cudaMemcpy3DParmsVal.srcArray;    // Either srcArray or srcPtr.
                    //cudaMemcpy3DParmsVal.srcPos;      // Optional. Offset in bytes.
                    cudaMemcpy3DParmsVal.srcPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBufSrc)),
                            alpaka::mem::getPitchBytes(memBufSrc),
                            uiSrcWidth,
                            uiSrcHeight);
                    //cudaMemcpy3DParmsVal.dstArray;    // Either dstArray or dstPtr.
                    //cudaMemcpy3DParmsVal.dstPos;      // Optional. Offset in bytes.
                    cudaMemcpy3DParmsVal.dstPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBufDst)),
                            alpaka::mem::getPitchBytes(memBufDst),
                            uiDstWidth,
                            uiDstHeight);
                    cudaMemcpy3DParmsVal.extent = 
                        make_cudaExtent(
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            uiExtentDepth);
                    cudaMemcpy3DParmsVal.kind = cudaMemcpyKindVal;

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy3D(
                            &cudaMemcpy3DParmsVall));
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::MemBufCuda.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufCuda dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetDim<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufCuda width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetWidth<
                cuda::detail::MemBufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getWidth(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_uiWidth;
                }
            };

            //#############################################################################
            //! The MemBufCuda height get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetHeight<
                cuda::detail::MemBufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getHeight(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_uiHeight;
                }
            };
            //#############################################################################
            //! The MemBufCuda depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                cuda::detail::MemBufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getDepth(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_uiDepth;
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufCuda memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemSpace<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                using type = alpaka::mem::MemSpaceCuda;
            };

            //#############################################################################
            //! The MemBufCuda memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemElem<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetMemBuf<
                TElem, TDim, alpaka::mem::MemSpaceCuda>
            {
                using type = host::detail::MemBufCuda<TElem, TDim>;
            };

            //#############################################################################
            //! The MemBufCuda native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                static TElem const * getNativePtr(
                    cuda::detail::MemBufCuda<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                static TElem * getNativePtr(
                    cuda::detail::MemBufCuda<TElem, TDim> & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The CUDA buffer pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                static std::size_t getPitchBytes(
                    cuda::detail::MemBufCuda<TElem, TDim> const & memPitch)
                {
                    return memPitch.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The CUDA 1D memory allocation trait specialization.
            //#############################################################################
            template<typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim1, 
                alpaka::mem::MemSpaceCuda>
            {
                template<
                    typename TExtents>
                static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim1> memAlloc(
                    TExtents const & extents)
                {
                    auto const uiWidth(extent::getWidth(extents));
                    assert(uiWidth>0);
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);

                    void * pBuffer;
                    ALPAKA_CUDA_CHECK(cudaMalloc(
                        &pBuffer, 
                        uiWidthBytes));
                    assert((pBuffer));

                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim1>(
                            pBuffer,
                            uiWidthBytes,
                            extents);
                }
            };

            //#############################################################################
            //! The CUDA 2D memory allocation trait specialization.
            //#############################################################################
            template<typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim2, 
                alpaka::mem::MemSpaceCuda>
            {
                template<
                    typename TExtents>
                static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim2> memAlloc(
                    TExtents const & extents)
                {
                    auto const uiWidth(extent::getWidth(extents));
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);
                    auto const uiHeight(extent::getHeight(extents));
                    auto const uiElementCount(uiWidth * uiHeight;
                    assert(uiElementCount>0);

                    void * pBuffer;
                    int iPitch;
                    ALPAKA_CUDA_CHECK(cudaMallocPitch(
                        &pBuffer,
                        &iPitch,
                        uiWidthBytes,
                        uiHeight));
                    assert(pBuffer);
                    assert(iPitch>uiWidthBytes);
                    
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim2>(
                            pBuffer,
                            static_cast<std::size_t>(iPitch),
                            extents);
                }
            };

            //#############################################################################
            //! The CUDA 3D memory allocation trait specialization.
            //#############################################################################
            template<typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim3, 
                alpaka::mem::MemSpaceCuda>
            {
                template<
                    typename TExtents>
                static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim3> memAlloc(
                    TExtents const & extents)
                {
                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            extent::getWidth(extents),
                            extent::getHeight(extents),
                            extent::getDepth(extents)));
                    
                    cudaPitchedPtr cudaPitchedPtrVal;
                    ALPAKA_CUDA_CHECK(cudaMalloc3D(
                        &cudaPitchedPtrVal,
                        cudaExtentVal));

                    assert(cudaPitchedPtrVal.ptr);
                    
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim3>(
                            cudaPitchedPtrVal.ptr,
                            static_cast<std::size_t>(cudaPitchedPtrVal.pitch),
                            extents);
                }
            };

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
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst,
                    TMemBufSrc const & memBufSrc,
                    TExtents const & extents)
                {
                    cuda::detail::pageLockHostMem(memBufDst);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyDeviceToHost);

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
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents)
                {
                    cuda::detail::pageLockHostMem(memBufSrc);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extents,
                        cudaMemcpyHostToDevice);

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
            };

            //#############################################################################
            //! The CUDA 1D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim1,
                alpaka::mem::MemSpaceCuda>
            {
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::Dim1>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim1 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);

                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf)),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBuf>)));
                }
            };
            //#############################################################################
            //! The CUDA 2D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim2,
                alpaka::mem::MemSpaceCuda>
            {
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::Dim2>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim2 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);

                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset2D(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf)),
                            alpaka::mem::getPitchBytes(memBuf),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBuf>),
                            uiExtentHeight));
                }
            };
            //#############################################################################
            //! The CUDA 3D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim3,
                alpaka::mem::MemSpaceCuda>
            {
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::Dim3>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim3 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf)),
                            alpaka::mem::getPitchBytes(memBuf),
                            uiDstWidth,
                            uiDstHeight));
                    
                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            uiExtentWidth,
                            uiExtentHeight,
                            uiExtentDepth));
                    
                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset3D(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal));
                }
            };
        }
    }
}
