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

#include <alpaka/cuda/mem/Space.hpp>    // SpaceCuda
#include <alpaka/cuda/mem/Set.hpp>      // Set
#include <alpaka/cuda/Common.hpp>

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>
#include <alpaka/core/Vec.hpp>          // Vec<TDim::value>

#include <alpaka/traits/mem/Buf.hpp>    // traits::Copy
#include <alpaka/traits/Extents.hpp>    // traits::getXXX

#include <cassert>                      // assert
#include <memory>                       // std::shared_ptr

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
            class BufCuda
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
                ALPAKA_FCT_HOST BufCuda(
                    TElem * const pMem,
                    UInt const & uiPitchBytes,
                    TExtents const & extents) :
                        m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
                        m_spMem(pMem, &BufCuda::freeBuffer),
                        m_uiPitchBytes(uiPitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::DimT<TExtents>>::value,
                        "The extents are required to have the same dimensionality as the BufCuda!");
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
                UInt m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for cuda::detail::BufCuda.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The BufCuda dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DimType<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The BufCuda extents get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetExtents<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<TDim::value> getExtents(
                    cuda::detail::BufCuda<TElem, TDim> const & extents)
                {
                    return {extents.m_vExtentsElements};
                }
            };

            //#############################################################################
            //! The BufCuda width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetWidth<
                cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getWidth(
                    cuda::detail::BufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The BufCuda height get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetHeight<
                cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getHeight(
                    cuda::detail::BufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The BufCuda depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getDepth(
                    cuda::detail::BufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }
        
        namespace offset
        {
            //#############################################################################
            //! The BufCuda offsets get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetOffsets<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<TDim::value> getOffsets(
                    cuda::detail::BufCuda<TElem, TDim> const &)
                {
                    return Vec<TDim::value>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The BufCuda base memory buffer trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct IsBufBase<
                cuda::detail::BufCuda<TElem, TDim>> :
                    std::true_type
            {};

            //#############################################################################
            //! The BufCuda memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct SpaceType<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = alpaka::mem::SpaceCuda;
            };

            //#############################################################################
            //! The BufCuda memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct ElemType<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The BufCuda base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetBuf<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static cuda::detail::BufCuda<TElem, TDim> const & getBuf(
                    cuda::detail::BufCuda<TElem, TDim> const & buf)
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static cuda::detail::BufCuda<TElem, TDim> & getBuf(
                    cuda::detail::BufCuda<TElem, TDim> & buf)
                {
                    return buf;
                }
            };

            //#############################################################################
            //! The BufCuda native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem const * getNativePtr(
                    cuda::detail::BufCuda<TElem, TDim> const & buf)
                {
                    return buf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem * getNativePtr(
                    cuda::detail::BufCuda<TElem, TDim> & buf)
                {
                    return buf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The CUDA buffer pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getPitchBytes(
                    cuda::detail::BufCuda<TElem, TDim> const & pitch)
                {
                    return pitch.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The CUDA 1D memory allocation trait specialization.
            //#############################################################################
            template<
                typename T>
            struct Alloc<
                T, 
                alpaka::dim::Dim1, 
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static alpaka::cuda::detail::BufCuda<T, alpaka::dim::Dim1> alloc(
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
                        alpaka::cuda::detail::BufCuda<T, alpaka::dim::Dim1>(
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
            struct Alloc<
                T, 
                alpaka::dim::Dim2, 
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static alpaka::cuda::detail::BufCuda<T, alpaka::dim::Dim2> alloc(
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
                        alpaka::cuda::detail::BufCuda<T, alpaka::dim::Dim2>(
                            reinterpret_cast<T *>(pBuffer),
                            static_cast<UInt>(uiPitch),
                            extents);
                }
            };

            //#############################################################################
            //! The CUDA 3D memory allocation trait specialization.
            //#############################################################################
            template<
                typename T>
            struct Alloc<
                T, 
                alpaka::dim::Dim3, 
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static alpaka::cuda::detail::BufCuda<T, alpaka::dim::Dim3> alloc(
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
                        alpaka::cuda::detail::BufCuda<T, alpaka::dim::Dim3>(
                            reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                            static_cast<UInt>(cudaPitchedPtrVal.pitch),
                            extents);
                }
            };
        }
    }
}
