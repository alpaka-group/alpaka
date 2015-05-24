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

#include <boost/predef.h>           // workarounds

#include <cstddef>                  // std::size_t
#include <type_traits>              // std::remove_cv, std::integral_constant

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Rounds to the next higher power of two (if not already power of two).
    // Adapted from llvm/ADT/SmallPtrSet.h
    //-----------------------------------------------------------------------------
    template<
        std::size_t N>
    struct RoundUpToPowerOfTwo;

    //-----------------------------------------------------------------------------
    //! Defines implementation details that should not be used directly by the user.
    //-----------------------------------------------------------------------------
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! Base case for N being a power of two.
        //-----------------------------------------------------------------------------
        template<
            std::size_t N,
            bool TbIsPowerTwo>
        struct RoundUpToPowerOfTwoHelper :
            std::integral_constant<
                std::size_t,
                N>
        {};
        //-----------------------------------------------------------------------------
        //! Case for N not being a power of two.
        // We could just use NextVal = N+1, but this converges faster.  N|(N-1) sets
        // the right-most zero bits to one all at once, e.g. 0b0011000 -> 0b0011111.
        //-----------------------------------------------------------------------------
        template<
            std::size_t N>
        struct RoundUpToPowerOfTwoHelper<
            N,
            false> :
                std::integral_constant<
                    std::size_t,
                    RoundUpToPowerOfTwo<(N | (N - 1)) + 1>::value>
        {};
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        std::size_t N>
    struct RoundUpToPowerOfTwo :
        std::integral_constant<
            std::size_t,
            detail::RoundUpToPowerOfTwoHelper<
                N,
                (N&(N - 1)) == 0>::value>
    {};

    //-----------------------------------------------------------------------------
    //! The alignment methods.
    //-----------------------------------------------------------------------------
    namespace align
    {
        //-----------------------------------------------------------------------------
        //! Calculates the optimal alignment for data of the given size.
        //-----------------------------------------------------------------------------
        template<
            std::size_t TuiSizeBytes>
        struct OptimalAlignment :
            std::integral_constant<
                std::size_t,
                // GCC does not support alignments larger then 128: "warning: requested alignment 256 is larger than 128[-Wattributes]".
                ((TuiSizeBytes > 64)
                    ? 128
                    : RoundUpToPowerOfTwo<TuiSizeBytes>::value)>
        {};
    }
}

// Newer GCC versions >= 4.9 do not support constant expressions as parameters to alignas: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58109
#if BOOST_COMP_GNUC || BOOST_COMP_INTEL
    #define ALPAKA_OPTIMAL_ALIGNMENT(uiSizeBytes)\
            ((uiSizeBytes)==1?1:\
            ((uiSizeBytes)<=2?2:\
            ((uiSizeBytes)<=4?4:\
            ((uiSizeBytes)<=8?8:\
            ((uiSizeBytes)<=16?16:\
            ((uiSizeBytes)<=32?32:\
            ((uiSizeBytes)<=64?64:128\
            )))))))
    //-----------------------------------------------------------------------------
    //! Aligns the data optimally.
    //! You must align all arrays and structs which can be used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN(TYPE, NAME) alignas(ALPAKA_OPTIMAL_ALIGNMENT(sizeof(typename std::remove_cv<TYPE>::type))) TYPE NAME
    //-----------------------------------------------------------------------------
    //! Aligns the data at 8 bytes.
    //! You must align all arrays and structs which can be used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN_8(TYPE, NAME) alignas(8) TYPE NAME

    //-----------------------------------------------------------------------------
    //! \return The alignment of the type.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGNOF(TYPE) alignof(TYPE)
#elif (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
    //-----------------------------------------------------------------------------
    //! Aligns the data optimally.
    //! You must align all arrays and structs which can be used on accelerators.
    //-----------------------------------------------------------------------------
    // \FIXME: sizeof in __declspec(align( not allowed...
    //#define ALPAKA_ALIGN(TYPE, NAME) __declspec(align(ALPAKA_OPTIMAL_ALIGNMENT(sizeof(typename std::remove_cv<TYPE>::type)))) TYPE NAME
    #define ALPAKA_ALIGN(TYPE, NAME) __declspec(align(16)) TYPE NAME
    //-----------------------------------------------------------------------------
    //! Aligns the data at 8 bytes.
    //! You must align all arrays and structs which can be used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN_8(TYPE, NAME) __declspec(align(8)) TYPE NAME

    //-----------------------------------------------------------------------------
    //! \return The alignment of the type.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGNOF(TYPE) __alignof(TYPE)
#else
    //-----------------------------------------------------------------------------
    //! Aligns the data optimally.
    //! You must align all arrays and structs which can be used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN(TYPE, NAME) alignas(alpaka::align::OptimalAlignment<sizeof(typename std::remove_cv<TYPE>::type)>::value) TYPE NAME
    //-----------------------------------------------------------------------------
    //! Aligns the data at 8 bytes.
    //! You must align all arrays and structs which can be used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN_8(TYPE, NAME) alignas(8) TYPE NAME

    //-----------------------------------------------------------------------------
    //! \return The alignment of the type.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGNOF(TYPE) alignof(TYPE)
#endif
