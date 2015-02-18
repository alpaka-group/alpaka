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

#include <boost/predef.h>               // workarounds

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
            //! The fixed size array extents get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetExtents<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static /*constexpr*/ Vec<alpaka::dim::DimT<TFixedSizeArray>::value> getExtents(
                    TFixedSizeArray const & extents)
                {
                    return getExtentsInternal(
                        extents,
                        IdxSequence());
                }

            private:
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                using IdxSequence = typename alpaka::detail::make_index_sequence<std::rank<TFixedSizeArray>::value>::type;
#else
                using IdxSequence = alpaka::detail::make_index_sequence<std::rank<TFixedSizeArray>::value>;
#endif
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TVal,
                    UInt... TIndices>
                ALPAKA_FCT_HOST static /*constexpr*/ Vec<alpaka::dim::DimT<TFixedSizeArray>::value> getExtentsInternal(
                    TFixedSizeArray const & extents,
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
                alpaka::detail::index_sequence<TIndices...> const &)
#else
                alpaka::detail::integer_sequence<UInt, TIndices...> const &)
#endif
                {
                    return {(std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value-(TIndices+1u)>::value)...};
                }
            };

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
                ALPAKA_FCT_HOST_ACC static constexpr UInt getWidth(
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
                ALPAKA_FCT_HOST_ACC static constexpr UInt getHeight(
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
                ALPAKA_FCT_HOST_ACC static constexpr UInt getDepth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 3u>::value;
                }
            };
        }
        
        namespace offset
        {
            //#############################################################################
            //! The fixed size array offsets get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetOffsets<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<alpaka::dim::DimT<TFixedSizeArray>::value> getOffsets(
                    TFixedSizeArray const &)
                {
                    return Vec<alpaka::dim::DimT<TFixedSizeArray>::value>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The fixed size array base memory buffer trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct IsMemBufBase<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                static const bool value = true;
            };

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
                ALPAKA_FCT_HOST static TFixedSizeArray const & getMemBufBase(
                    TFixedSizeArray const & memBufBase)
                {
                    return memBufBase;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TFixedSizeArray & getMemBufBase(
                    TFixedSizeArray & memBufBase)
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

                ALPAKA_FCT_HOST_ACC static constexpr UInt getPitchBytes(
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
                UInt TuiSize>
            struct DimType<
                std::array<TElem, TuiSize>>
            {
                using type = alpaka::dim::Dim1;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The std::array extents get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct GetExtents<
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static /*constexpr*/ Vec<1u> getExtents(
                    std::array<TElem, TuiSize> const & extents)
                {
                    return {TuiSize};
                }
            };

            //#############################################################################
            //! The std::array width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct GetWidth<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST_ACC static constexpr UInt getWidth(
                    std::array<TElem, TuiSize> const & extent)
                {
                    return TuiSize;
                }
            };
        }
        
        namespace offset
        {
            //#############################################################################
            //! The std::array offsets get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct GetOffsets<
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<1u> getOffsets(
                    std::array<TElem, TuiSize> const &)
                {
                    return Vec<1u>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The std::array base memory buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct IsMemBufBase<
                std::array<TElem, TuiSize>>
            {
                static const bool value = true;
            };

            //#############################################################################
            //! The std::array memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
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
                UInt TuiSize>
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
                UInt TuiSize>
            struct GetMemBufBase<
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::array<TElem, TuiSize> const & getMemBufBase(
                    std::array<TElem, TuiSize> const & memBufBase)
                {
                    return memBufBase;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::array<TElem, TuiSize> & getMemBufBase(
                    std::array<TElem, TuiSize> & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The std::array native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
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
                UInt TuiSize>
            struct GetPitchBytes<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST_ACC static UInt getPitchBytes(
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
                typename TAllocator>
            struct DimType<
                std::vector<TElem, TAllocator>>
            {
                using type = alpaka::dim::Dim1;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The std::vector extents get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetExtents<
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<1u> getExtents(
                    std::vector<TElem, TAllocator> const & extent)
                {
                    return {extent.size()};
                }
            };

            //#############################################################################
            //! The std::vector width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetWidth<
                std::vector<TElem, TAllocator>>
            {
                ALPAKA_FCT_HOST_ACC static UInt getWidth(
                    std::vector<TElem, TAllocator> const & extent)
                {
                    return extent.size();
                }
            };
        }
        
        namespace offset
        {
            //#############################################################################
            //! The std::vector offsets get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetOffsets<
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<1u> getOffsets(
                    std::vector<TElem, TAllocator> const &)
                {
                    return Vec<1u>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The std::vector base memory buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct IsMemBufBase<
                std::vector<TElem, TAllocator>>
            {
                static const bool value = true;
            };

            //#############################################################################
            //! The std::vector memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct MemSpaceType<
                std::vector<TElem, TAllocator>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The std::vector memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct MemElemType<
                std::vector<TElem, TAllocator>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The std::vector base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetMemBufBase<
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::vector<TElem, TAllocator> const & getMemBufBase(
                    std::vector<TElem, TAllocator> const & memBufBase)
                {
                    return memBufBase;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::vector<TElem, TAllocator> & getMemBufBase(
                    std::vector<TElem, TAllocator> & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The std::vector native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetNativePtr<
                std::vector<TElem, TAllocator>>
            {
                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    std::vector<TElem, TAllocator> const & memBuf)
                {
                    return memBuf.data();
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    std::vector<TElem, TAllocator> & memBuf)
                {
                    return memBuf.data();
                }
            };

            //#############################################################################
            //! The std::vector pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetPitchBytes<
                std::vector<TElem, TAllocator>>
            {
                ALPAKA_FCT_HOST_ACC static UInt getPitchBytes(
                    std::vector<TElem, TAllocator> const & memPitch)
                {
                    return sizeof(TElem) * memPitch.size();
                }
            };
        }
    }
}
