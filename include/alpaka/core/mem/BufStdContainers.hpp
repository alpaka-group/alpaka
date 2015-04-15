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

#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST

#include <alpaka/traits/Dim.hpp>            // dim::DimType
#include <alpaka/traits/Extent.hpp>         // traits::getXXX
#include <alpaka/traits/Mem.hpp>            // mem::SpaceType

#include <alpaka/host/mem/Space.hpp>        // SpaceHost
#include <alpaka/accs/cuda/mem/Space.hpp>   // SpaceCuda
#include <alpaka/host/Dev.hpp>              // host::getDev

#include <boost/predef.h>                   // workarounds

#include <type_traits>                      // std::enable_if, std::is_array, std::extent
#include <vector>                           // std::vector
#include <array>                            // std::array

namespace alpaka
{
    namespace accs
    {
        namespace serial
        {
            namespace detail
            {
                class DevSerial;
            }
        }
    }
}

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
        namespace dev
        {
            //#############################################################################
            //! The fixed size array device type trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct DevType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = accs::serial::detail::DevSerial;
            };

            //#############################################################################
            //! The fixed size array device get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetDev<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    TFixedSizeArray const & buf)
                    -> accs::serial::detail::DevSerial
                {
                    return alpaka::host::getDev();
                }
            };
        }

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
                ALPAKA_FCT_HOST static /*constexpr*/ auto getExtents(
                    TFixedSizeArray const & extents)
                -> Vec<alpaka::dim::DimT<TFixedSizeArray>::value>
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
                    size_t... TIndices>
                ALPAKA_FCT_HOST static /*constexpr*/ auto getExtentsInternal(
                    TFixedSizeArray const & extents,
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
                alpaka::detail::index_sequence<TIndices...> const &)
#else
                alpaka::detail::integer_sequence<std::size_t, TIndices...> const &)
#endif
                -> Vec<alpaka::dim::DimT<TFixedSizeArray>::value>
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
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                ALPAKA_FCT_HOST_ACC static auto getWidth(
#else
                ALPAKA_FCT_HOST_ACC static constexpr auto getWidth(
#endif
                    TFixedSizeArray const &)
                -> UInt
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
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                ALPAKA_FCT_HOST_ACC static auto getHeight(
#else
                ALPAKA_FCT_HOST_ACC static constexpr auto getHeight(
#endif
                    TFixedSizeArray const &)
                -> UInt
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
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                ALPAKA_FCT_HOST_ACC static auto getDepth(
#else
                ALPAKA_FCT_HOST_ACC static constexpr auto getDepth(
#endif
                    TFixedSizeArray const &)
                -> UInt
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
                ALPAKA_FCT_HOST static auto getOffsets(
                    TFixedSizeArray const &)
                -> Vec<alpaka::dim::DimT<TFixedSizeArray>::value>
                {
                    return Vec<alpaka::dim::DimT<TFixedSizeArray>::value>(0u);
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
            struct SpaceType<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
#ifdef __CUDA_ARCH__
                using type = alpaka::mem::SpaceCuda;
#else
                using type = alpaka::mem::SpaceHost;
#endif
            };

            //#############################################################################
            //! The fixed size array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct ElemType<
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
            struct GetBuf<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    TFixedSizeArray const & buf)
                -> TFixedSizeArray const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    TFixedSizeArray & buf)
                -> TFixedSizeArray &
                {
                    return buf;
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

                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    TFixedSizeArray const & buf)
                -> TElem const *
                {
                    return buf;
                }
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    TFixedSizeArray & buf)
                -> TElem *
                {
                    return buf;
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

#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
#else
                ALPAKA_FCT_HOST_ACC static constexpr auto getPitchBytes(
#endif
                    TFixedSizeArray const &)
                -> UInt
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
        namespace dev
        {
            //#############################################################################
            //! The std::array device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct DevType<
                std::array<TElem, TuiSize>>
            {
                using type = accs::serial::detail::DevSerial;
            };

            //#############################################################################
            //! The std::array device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct GetDev<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    std::array<TElem, TuiSize> const & buf)
                    -> accs::serial::detail::DevSerial
                {
                    return alpaka::host::getDev();
                }
            };
        }

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
                ALPAKA_FCT_HOST static /*constexpr*/ auto getExtents(
                    std::array<TElem, TuiSize> const & extents)
                -> Vec<1u>
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
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                ALPAKA_FCT_HOST_ACC static auto getWidth(
#else
                ALPAKA_FCT_HOST_ACC static constexpr auto getWidth(
#endif
                    std::array<TElem, TuiSize> const & extent)
                -> UInt
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
                ALPAKA_FCT_HOST static auto getOffsets(
                    std::array<TElem, TuiSize> const &)
                -> Vec<1u>
                {
                    return Vec<1u>(0u);
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
                UInt TuiSize>
            struct SpaceType<
                std::array<TElem, TuiSize>>
            {
                using type = alpaka::mem::SpaceHost;
            };

            //#############################################################################
            //! The std::array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                UInt TuiSize>
            struct ElemType<
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
            struct GetBuf<
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    std::array<TElem, TuiSize> const & buf)
                -> std::array<TElem, TuiSize> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    std::array<TElem, TuiSize> & buf)
                -> std::array<TElem, TuiSize> &
                {
                    return buf;
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
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    std::array<TElem, TuiSize> const & buf)
                -> TElem const *
                {
                    return buf.data();
                }
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    std::array<TElem, TuiSize> & buf)
                -> TElem *
                {
                    return buf.data();
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
                ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                    std::array<TElem, TuiSize> const & pitch)
                -> UInt
                {
                    return sizeof(TElem) * pitch.size();
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::vector.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The std::vector device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct DevType<
                std::vector<TElem, TAllocator>>
            {
                using type = accs::serial::detail::DevSerial;
            };

            //#############################################################################
            //! The std::vector device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetDev<
                std::vector<TElem, TAllocator>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    std::vector<TElem, TAllocator> const & buf)
                    -> accs::serial::detail::DevSerial
                {
                    return alpaka::host::getDev();
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The std::vector dimension getter trait specialization.
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
                ALPAKA_FCT_HOST static auto getExtents(
                    std::vector<TElem, TAllocator> const & extent)
                -> Vec<1u>
                {
                    return {static_cast<Vec<1u>::Val>(extent.size())};
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
                ALPAKA_FCT_HOST_ACC static auto getWidth(
                    std::vector<TElem, TAllocator> const & extent)
                -> UInt
                {
                    return static_cast<UInt>(extent.size());
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
                ALPAKA_FCT_HOST static auto getOffsets(
                    std::vector<TElem, TAllocator> const &)
                -> Vec<1u>
                {
                    return Vec<1u>(0u);
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
                typename TAllocator>
            struct SpaceType<
                std::vector<TElem, TAllocator>>
            {
                using type = alpaka::mem::SpaceHost;
            };

            //#############################################################################
            //! The std::vector memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct ElemType<
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
            struct GetBuf<
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    std::vector<TElem, TAllocator> const & buf)
                -> std::vector<TElem, TAllocator> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    std::vector<TElem, TAllocator> & buf)
                -> std::vector<TElem, TAllocator> &
                {
                    return buf;
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
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    std::vector<TElem, TAllocator> const & buf)
                -> TElem const *
                {
                    return buf.data();
                }
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    std::vector<TElem, TAllocator> & buf)
                -> TElem *
                {
                    return buf.data();
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
                ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                    std::vector<TElem, TAllocator> const & pitch)
                -> UInt
                {
                    return static_cast<UInt>(sizeof(TElem) * pitch.size());
                }
            };
        }
    }
}
