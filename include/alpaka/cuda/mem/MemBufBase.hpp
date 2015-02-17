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

#include <alpaka/cuda/MemSpace.hpp>         // MemSpaceCuda
#include <alpaka/cuda/mem/MemSet.hpp>       // MemSet
#include <alpaka/cuda/Common.hpp>

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Vec.hpp>              // Vec<TDim::value>

#include <alpaka/traits/mem/MemBufBase.hpp> // traits::MemCopy
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

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
            class MemBufBaseCuda
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
                ALPAKA_FCT_HOST MemBufBaseCuda(
                    TElem * const pMem,
                    std::size_t const & uiPitchBytes,
                    TExtents const & extents) :
                        m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
                        m_spMem(pMem, &MemBufBaseCuda::freeBuffer),
                        m_uiPitchBytes(uiPitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::DimT<TExtents>>::value,
                        "The extents are required to have the same dimensionality as the MemBufBaseCuda!");
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
                Vec<TDim::value> m_vExtentsElements;
                std::shared_ptr<TElem> m_spMem;
                std::size_t m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for cuda::detail::MemBufBaseCuda.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufBaseCuda dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DimType<
                cuda::detail::MemBufBaseCuda<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufBaseCuda width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetWidth<
                cuda::detail::MemBufBaseCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getWidth(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The MemBufBaseCuda height get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetHeight<
                cuda::detail::MemBufBaseCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getHeight(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The MemBufBaseCuda depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                cuda::detail::MemBufBaseCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDepth(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufBaseCuda memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct MemSpaceType<
                cuda::detail::MemBufBaseCuda<TElem, TDim>>
            {
                using type = alpaka::mem::MemSpaceCuda;
            };

            //#############################################################################
            //! The MemBufBaseCuda memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct MemElemType<
                cuda::detail::MemBufBaseCuda<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufBaseCuda base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetMemBufBase<
                cuda::detail::MemBufBaseCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static cuda::detail::MemBufBaseCuda<TElem, TDim> getMemBufBase(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> const & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The MemBufBaseCuda native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                cuda::detail::MemBufBaseCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem const * getNativePtr(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem * getNativePtr(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> & memBuf)
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
                cuda::detail::MemBufBaseCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getPitchBytes(
                    cuda::detail::MemBufBaseCuda<TElem, TDim> const & memPitch)
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
                ALPAKA_FCT_HOST static alpaka::cuda::detail::MemBufBaseCuda<T, alpaka::dim::Dim1> memAlloc(
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
                        alpaka::cuda::detail::MemBufBaseCuda<T, alpaka::dim::Dim1>(
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
                ALPAKA_FCT_HOST static alpaka::cuda::detail::MemBufBaseCuda<T, alpaka::dim::Dim2> memAlloc(
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
                        alpaka::cuda::detail::MemBufBaseCuda<T, alpaka::dim::Dim2>(
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
                ALPAKA_FCT_HOST static alpaka::cuda::detail::MemBufBaseCuda<T, alpaka::dim::Dim3> memAlloc(
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
                        alpaka::cuda::detail::MemBufBaseCuda<T, alpaka::dim::Dim3>(
                            reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                            static_cast<std::size_t>(cudaPitchedPtrVal.pitch),
                            extents);
                }
            };
        }
    }
}
