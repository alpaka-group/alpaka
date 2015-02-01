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

#include <alpaka/cuda/MemSpace.hpp>         // MemSpaceCuda
#include <alpaka/cuda/Stream.hpp>           // StreamCuda
#include <alpaka/cuda/Common.hpp>

#include <cstddef>                          // std::size_t
#include <cassert>                          // assert

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for MemSet.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The CUDA 1D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim1,
                alpaka::mem::MemSpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TMemBuf, 
                    typename TExtents>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
                        cudaMemsetAsync(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf)),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBuf>),
                            *stream.m_spCudaStream.get()));
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
                        cudaMemset2DAsync(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(memBuf)),
                            alpaka::mem::getPitchBytes(memBuf),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::GetMemElemT<TMemBuf>),
                            uiExtentHeight,
                            *stream.m_spCudaStream.get()));
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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

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
                        cudaMemset3DAsync(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal,
                            *stream.m_spCudaStream.get()));
                }
            };
        }
    }
}
