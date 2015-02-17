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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

#include <alpaka/traits/Dim.hpp>        // dim::DimType
#include <alpaka/traits/Extents.hpp>    // traits::getXXX
#include <alpaka/traits/Mem.hpp>        // mem::MemSpaceType

#include <alpaka/host/MemSpace.hpp>     // MemSpaceHost
#include <alpaka/cuda/MemSpace.hpp>     // MemSpaceCuda

#include <type_traits>                  // std::enable_if, std::is_array, std::extent
#include <vector>                       // std::vector
#include <array>                        // std::array

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for fixed size arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory ops.
    // Up to 3 dimensions are supported.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The fixed size array dimension getter trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct DimType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = alpaka::dim::Dim<std::rank<TFixedSizeArray>::value>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The fixed size array width get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetWidth<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value >= 1u)
                    && (std::rank<TFixedSizeArray>::value <= 3u)
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value-1u>::value;
                }
            };

            //#############################################################################
            //! The fixed size array height get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetHeight<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value >= 2u)
                    && (std::rank<TFixedSizeArray>::value <= 3u)
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 2u>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getHeight(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 2u>::value;
                }
            };
            //#############################################################################
            //! The fixed size array depth get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetDepth<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value >= 3u)
                    && (std::rank<TFixedSizeArray>::value <= 3u)
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 3u>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getDepth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 3u>::value;
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The fixed size array memory space trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct MemSpaceType<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
#ifdef __CUDA_ARCH__
                using type = alpaka::mem::MemSpaceCuda;
#else
                using type = alpaka::mem::MemSpaceHost;
#endif
            };

            //#############################################################################
            //! The fixed size array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct MemElemType<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = typename std::remove_all_extents<TFixedSizeArray>::type;
            };

            //#############################################################################
            //! The fixed size array base buffer trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetMemBufBase<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TFixedSizeArray getMemBufBase(
                    TFixedSizeArray const & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The fixed size array native pointer get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetNativePtr<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    TFixedSizeArray const & memBuf)
                {
                    return memBuf;
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    TFixedSizeArray & memBuf)
                {
                    return memBuf;
                }
            };

            //#############################################################################
            //! The fixed size array pitch get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetPitchBytes<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
            {
                using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                ALPAKA_FCT_HOST_ACC static std::size_t getPitchBytes(
                    TFixedSizeArray const &)
                {
                    return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::array.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The std::array dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct DimType<
                std::array<TElem, TuiSize>>
            {
                using type = alpaka::dim::Dim1;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The std::array width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct GetWidth<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    std::array<TElem, TuiSize> const & extent)
                {
                    return extent.size();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The std::array memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct MemSpaceType<
                std::array<TElem, TuiSize>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The std::array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct MemElemType<
                std::array<TElem, TuiSize>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The std::array base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct GetMemBufBase<
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::array<TElem, TuiSize> getMemBufBase(
                    std::array<TElem, TuiSize> const & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The std::array native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct GetNativePtr<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    std::array<TElem, TuiSize> const & memBuf)
                {
                    return memBuf.data();
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    std::array<TElem, TuiSize> & memBuf)
                {
                    return memBuf.data();
                }
            };

            //#############################################################################
            //! The std::array pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct GetPitchBytes<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getPitchBytes(
                    std::array<TElem, TuiSize> const & memPitch)
                {
                    return sizeof(TElem) * memPitch.size();
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::vector.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The dimension getter trait.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct DimType<
                std::vector<TElem, Allocator>>
            {
                using type = alpaka::dim::Dim1;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The std::vector width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetWidth<
                std::vector<TElem, Allocator>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    std::vector<TElem, Allocator> const & extent)
                {
                    return extent.size();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The std::vector memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct MemSpaceType<
                std::vector<TElem, Allocator>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The std::vector memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct MemElemType<
                std::vector<TElem, Allocator>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The std::vector base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetMemBufBase<
                std::vector<TElem, Allocator>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::vector<TElem, Allocator> getMemBufBase(
                    std::vector<TElem, Allocator> const & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The std::vector native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetNativePtr<
                std::vector<TElem, Allocator>>
            {
                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    std::vector<TElem, Allocator> const & memBuf)
                {
                    return memBuf.data();
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    std::vector<TElem, Allocator> & memBuf)
                {
                    return memBuf.data();
                }
            };

            //#############################################################################
            //! The std::vector pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetPitchBytes<
                std::vector<TElem, Allocator>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getPitchBytes(
                    std::vector<TElem, Allocator> const & memPitch)
                {
                    return sizeof(TElem) * memPitch.size();
                }
            };
        }
    }
}
