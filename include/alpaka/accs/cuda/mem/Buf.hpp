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

#include <alpaka/accs/cuda/mem/Space.hpp>   // SpaceCuda
#include <alpaka/accs/cuda/mem/Set.hpp>     // Set
#include <alpaka/accs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/accs/cuda/Common.hpp>

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Vec.hpp>              // Vec<TDim::value>

#include <alpaka/traits/mem/Buf.hpp>        // traits::Copy
#include <alpaka/traits/Extent.hpp>         // traits::getXXX

#include <cassert>                          // assert
#include <memory>                           // std::shared_ptr

namespace alpaka
{
    namespace accs
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
                        DevCuda dev,
                        TElem * const pMem,
                        UInt const & uiPitchBytes,
                        TExtents const & extents) :
                            m_Dev(dev),
                            m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
                            m_spMem(
                                pMem, 
                                std::bind(&BufCuda::freeBuffer, std::placeholders::_1, std::ref(m_Dev))),
                            m_uiPitchBytes(uiPitchBytes)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            TDim::value == dim::DimT<TExtents>::value,
                            "The extents are required to have the same dimensionality as the BufCuda!");
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Frees the shared buffer.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto freeBuffer(
                        TElem * pBuffer,
                        DevCuda const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        assert(pBuffer);

                        // Set the current device. \TODO: Is setting the current device before cudaFree required?
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev.m_iDevice));
                        // Free the buffer.
                        cudaFree(reinterpret_cast<void *>(pBuffer));
                    }

                public:
                    DevCuda m_Dev;
                    Vec<TDim::value> m_vExtentsElements;
                    std::shared_ptr<TElem> m_spMem;
                    UInt m_uiPitchBytes;
                };
            }
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCuda.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The BufCuda device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DevType<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = accs::cuda::detail::DevCuda;
            };

            //#############################################################################
            //! The BufCuda device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetDev<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> accs::cuda::detail::DevCuda
                {
                    return buf.m_Dev;
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The BufCuda dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DimType<
                accs::cuda::detail::BufCuda<TElem, TDim>>
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtents(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & extents)
                -> Vec<TDim::value>
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
                accs::cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getWidth(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & extent)
                -> UInt
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
                accs::cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getHeight(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & extent)
                -> UInt
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
                accs::cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getDepth(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & extent)
                -> UInt
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffsets(
                    accs::cuda::detail::BufCuda<TElem, TDim> const &)
                -> Vec<TDim::value>
                {
                    return Vec<TDim::value>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The BufCuda memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct SpaceType<
                accs::cuda::detail::BufCuda<TElem, TDim>>
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> accs::cuda::detail::BufCuda<TElem, TDim> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    accs::cuda::detail::BufCuda<TElem, TDim> & buf)
                -> accs::cuda::detail::BufCuda<TElem, TDim> &
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getNativePtr(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> TElem const *
                {
                    return buf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getNativePtr(
                    accs::cuda::detail::BufCuda<TElem, TDim> & buf)
                -> TElem *
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & pitch)
                -> UInt
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
                accs::cuda::detail::DevCuda,
                T,
                alpaka::dim::Dim1,
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    accs::cuda::detail::DevCuda const & dev,
                    TExtents const & extents)
                -> accs::cuda::detail::BufCuda<T, alpaka::dim::Dim1>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const uiWidth(alpaka::extent::getWidth(extents));
                    assert(uiWidth>0);
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    // Allocate the buffer on this device.
                    void * pBuffer;
                    ALPAKA_CUDA_RT_CHECK(cudaMalloc(
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
                        accs::cuda::detail::BufCuda<T, alpaka::dim::Dim1>(
                            dev,
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
                accs::cuda::detail::DevCuda,
                T,
                alpaka::dim::Dim2,
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    accs::cuda::detail::DevCuda const & dev,
                    TExtents const & extents)
                -> accs::cuda::detail::BufCuda<T, alpaka::dim::Dim2>
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

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    // Allocate the buffer on this device.
                    void * pBuffer;
                    std::size_t uiPitch;
                    ALPAKA_CUDA_RT_CHECK(cudaMallocPitch(
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
                        accs::cuda::detail::BufCuda<T, alpaka::dim::Dim2>(
                            dev,
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
                accs::cuda::detail::DevCuda,
                T,
                alpaka::dim::Dim3,
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    accs::cuda::detail::DevCuda const & dev,
                    TExtents const & extents)
                -> accs::cuda::detail::BufCuda<T, alpaka::dim::Dim3>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            alpaka::extent::getWidth(extents) * sizeof(T),
                            alpaka::extent::getHeight(extents),
                            alpaka::extent::getDepth(extents)));

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    // Allocate the buffer on this device.
                    cudaPitchedPtr cudaPitchedPtrVal;
                    ALPAKA_CUDA_RT_CHECK(cudaMalloc3D(
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
                        accs::cuda::detail::BufCuda<T, alpaka::dim::Dim3>(
                            dev,
                            reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                            static_cast<UInt>(cudaPitchedPtrVal.pitch),
                            extents);
                }
            };
        }
    }
}
