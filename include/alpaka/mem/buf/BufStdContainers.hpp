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

#include <alpaka/mem/buf/Traits.hpp>    // DevType, DimType, GetExtent,Copy, GetOffset, ...

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

// FIXME: This include can lead to circular include problems!
#include <alpaka/dev/DevCpu.hpp>        // DevCpu

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused
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
    namespace dev
    {
        namespace traits
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
                using type = dev::DevCpu;
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
                -> dev::DevCpu
                {
                    // \FIXME: CUDA device?
                    return dev::cpu::getDev();
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
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
                using type = dim::Dim<std::rank<TFixedSizeArray>::value>;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed size array width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TFixedSizeArray>
            struct GetExtent<
                TIdx,
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value > TIdx::value)
                    && (std::extent<TFixedSizeArray, TIdx::value>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static constexpr auto getExtent(
                    TFixedSizeArray const & /*extents*/)
                -> Uint
                {
                    // C++14
                    /*boost::ignore_unused(extents);*/
                    return std::extent<TFixedSizeArray, TIdx::value>::value;
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
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
                //! The fixed size array buf trait specialization.
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
                struct GetPtrNative<
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value>::type>
                {
                    using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                        TFixedSizeArray const & buf)
                    -> TElem const *
                    {
                        return buf;
                    }
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                        TFixedSizeArray & buf)
                    -> TElem *
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The fixed size array pitch get trait specialization.
                //#############################################################################
                // \FIXME: Calculate pitch for all dimensions, not only the last one.
                template<
                    typename TFixedSizeArray>
                struct GetPitchBytes<
                    std::integral_constant<Uint, std::rank<TFixedSizeArray>::value - 1u>,
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value
                        && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
                {
                    using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                    ALPAKA_FCT_HOST_ACC static constexpr auto getPitchBytes(
                        TFixedSizeArray const &)
                    -> Uint
                    {
                        return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed size array offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TFixedSizeArray>
            struct GetOffset<
                TIdx,
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    TFixedSizeArray const &)
                -> Uint
                {
                    return 0u;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::array.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                Uint TuiSize>
            struct DevType<
                std::array<TElem, TuiSize>>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The std::array device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                Uint TuiSize>
            struct GetDev<
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    std::array<TElem, TuiSize> const & buf)
                -> dev::DevCpu
                {
                    return dev::cpu::getDev();
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                Uint TuiSize>
            struct DimType<
                std::array<TElem, TuiSize>>
            {
                using type = dim::Dim1;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                Uint TuiSize>
            struct GetExtent<
                std::integral_constant<Uint, 0u>,
                std::array<TElem, TuiSize>>
            {
                ALPAKA_FCT_HOST_ACC static constexpr auto getExtent(
                    std::array<TElem, TuiSize> const & /*extents*/)
                -> Uint
                {
                    // C++14
                    /*boost::ignore_unused(extents);*/
                    return TuiSize;
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The std::array memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    Uint TuiSize>
                struct ElemType<
                    std::array<TElem, TuiSize>>
                {
                    using type = TElem;
                };

                //#############################################################################
                //! The std::array buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    Uint TuiSize>
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
                    Uint TuiSize>
                struct GetPtrNative<
                    std::array<TElem, TuiSize>>
                {
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                        std::array<TElem, TuiSize> const & buf)
                    -> TElem const *
                    {
                        return buf.data();
                    }
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
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
                    Uint TuiSize>
                struct GetPitchBytes<
                    std::integral_constant<Uint, 0u>,
                    std::array<TElem, TuiSize>>
                {
                    ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                        std::array<TElem, TuiSize> const & pitch)
                    -> Uint
                    {
                        return sizeof(TElem) * pitch.size();
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                Uint TuiSize>
            struct GetOffset<
                TIdx,
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    std::array<TElem, TuiSize> const &)
                -> Uint
                {
                    return 0u;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::vector.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
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
                using type = dev::DevCpu;
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
                -> dev::DevCpu
                {
                    return dev::cpu::getDev();
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
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
                using type = dim::Dim1;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetExtent<
                std::integral_constant<Uint, 0u>,
                std::vector<TElem, TAllocator>>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    std::vector<TElem, TAllocator> const & extents)
                -> Uint
                {
                    return static_cast<Uint>(extents.size());
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
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
                //! The std::vector buf trait specialization.
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
                struct GetPtrNative<
                    std::vector<TElem, TAllocator>>
                {
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                        std::vector<TElem, TAllocator> const & buf)
                    -> TElem const *
                    {
                        return buf.data();
                    }
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
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
                    std::integral_constant<Uint, 0u>,
                    std::vector<TElem, TAllocator>>
                {
                    ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                        std::vector<TElem, TAllocator> const & pitch)
                    -> Uint
                    {
                        return static_cast<Uint>(sizeof(TElem) * pitch.size());
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TAllocator>
            struct GetOffset<
                TIdx,
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    std::vector<TElem, TAllocator> const &)
                -> Uint
                {
                    return 0u;
                }
            };
        }
    }
}
