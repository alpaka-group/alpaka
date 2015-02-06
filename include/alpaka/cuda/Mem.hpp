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
#include <alpaka/core/Vec.hpp>              // Vec<TDim::value>

#include <alpaka/traits/Mem.hpp>            // traits::MemCopy
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <alpaka/cuda/MemSpace.hpp>         // MemSpaceCuda
#include <alpaka/cuda/mem/MemCopy.hpp>
#include <alpaka/cuda/mem/MemSet.hpp>
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
            class MemBufCuda
            {
            private:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST MemBufCuda(
                    TElem * const pMem,
                    std::size_t const & uiPitchBytes,
                    TExtents const & extents) :
                        m_vExtents(extents),
                        m_spMem(pMem, &MemBufCuda::freeBuffer),
                        m_uiPitchBytes(uiPitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::GetDimT<TExtents>>::value,
                        "The extents are required to have the same dimensionality as the MemBufCuda!");
                }

            private:
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void freeBuffer(
                    TElem * pBuffer)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    assert(pBuffer);
                    cudaFree(reinterpret_cast<void *>(pBuffer));
                }

            public:
                Vec<TDim::value> m_vExtents;
                std::shared_ptr<TElem> m_spMem;
                std::size_t m_uiPitchBytes; // \TODO: By using class specialization for Dim1 we could remove this value in this case.
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for cuda::detail::MemBufCuda.
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getWidth(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtents[0u];
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getHeight(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtents[1u];
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDepth(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtents[2u];
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
                using type = cuda::detail::MemBufCuda<TElem, TDim>;
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem const * getNativePtr(
                    cuda::detail::MemBufCuda<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem * getNativePtr(
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getPitchBytes(
                    cuda::detail::MemBufCuda<TElem, TDim> const & memPitch)
                {
                    return memPitch.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The CUDA 1D memory allocation trait specialization.
            //#############################################################################
            template<
                typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim1, 
                alpaka::mem::MemSpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim1> memAlloc(
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const uiWidth(alpaka::extent::getWidth(extents));
                    assert(uiWidth>0);
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);

                    void * pBuffer;
                    ALPAKA_CUDA_CHECK(cudaMalloc(
                        &pBuffer, 
                        uiWidthBytes));
                    assert((pBuffer));
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " ew: " << uiWidth
                        << " ewb: " << uiWidthBytes
                        << " ptr: " << pBuffer
                        << std::endl;
#endif
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim1>(
                            reinterpret_cast<T *>(pBuffer),
                            uiWidthBytes,
                            extents);
                }
            };

            //#############################################################################
            //! The CUDA 2D memory allocation trait specialization.
            //#############################################################################
            template<
                typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim2, 
                alpaka::mem::MemSpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim2> memAlloc(
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const uiWidth(alpaka::extent::getWidth(extents));
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);
                    auto const uiHeight(alpaka::extent::getHeight(extents));
#ifndef NDEBUG
                    auto const uiElementCount(uiWidth * uiHeight);
#endif
                    assert(uiElementCount>0);

                    void * pBuffer;
                    std::size_t uiPitch;
                    ALPAKA_CUDA_CHECK(cudaMallocPitch(
                        &pBuffer,
                        &uiPitch,
                        uiWidthBytes,
                        uiHeight));
                    assert(pBuffer);
                    assert(uiPitch>=uiWidthBytes);
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " ew: " << uiWidth
                        << " eh: " << uiHeight
                        << " ewb: " << uiWidthBytes
                        << " ptr: " << pBuffer
                        << " pitch: " << uiPitch
                        << std::endl;
#endif
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim2>(
                            reinterpret_cast<T *>(pBuffer),
                            uiPitch,
                            extents);
                }
            };

            //#############################################################################
            //! The CUDA 3D memory allocation trait specialization.
            //#############################################################################
            template<
                typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim3, 
                alpaka::mem::MemSpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim3> memAlloc(
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            alpaka::extent::getWidth(extents) * sizeof(T),
                            alpaka::extent::getHeight(extents),
                            alpaka::extent::getDepth(extents)));
                    
                    cudaPitchedPtr cudaPitchedPtrVal;
                    ALPAKA_CUDA_CHECK(cudaMalloc3D(
                        &cudaPitchedPtrVal,
                        cudaExtentVal));

                    assert(cudaPitchedPtrVal.ptr);
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " ew: " << alpaka::extent::getWidth(extents)
                        << " eh: " << cudaExtentVal.height
                        << " ed: " << cudaExtentVal.depth
                        << " ewb: " << cudaExtentVal.width
                        << " ptr: " << cudaPitchedPtrVal.ptr
                        << " pitch: " << cudaPitchedPtrVal.pitch
                        << " wb: " << cudaPitchedPtrVal.xsize
                        << " h: " << cudaPitchedPtrVal.ysize
                        << std::endl;
#endif
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim3>(
                            reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                            static_cast<std::size_t>(cudaPitchedPtrVal.pitch),
                            extents);
                }
            };
        }
    }
}
