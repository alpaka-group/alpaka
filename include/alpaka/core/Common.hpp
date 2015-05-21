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

#include <alpaka/core/Debug.hpp>

#include <boost/predef.h>           // workarounds

#include <type_traits>              // std::remove_cv
#include <cstddef>                  // std::size_t
#include <cstdint>                  // std::uint32_t

//-----------------------------------------------------------------------------
//! Disable nvcc warning:
//! calling a __host__ function from __host__ __device__ function.
//!
//! Usage:
//! ALPAKA_NO_HOST_ACC_WARNING
//! __device__ __host__ function_declaration()
//!
//! It is not possible to disable the warning for a __host__ function if there are calls of virtual functions inside.
//! For this case use a wrapper function.
//! WARNING: only use this method if there is no other way to create runnable code.
//! Most cases can solved by #ifdef __CUDA_ARCH__ or #ifdef __CUDACC__.
//-----------------------------------------------------------------------------
#if defined(ALPAKA_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #if BOOST_COMP_MSVC
        #define ALPAKA_NO_HOST_ACC_WARNING __pragma(hd_warning_disable)
    #else
        #define ALPAKA_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
    #endif
#else
    #define ALPAKA_NO_HOST_ACC_WARNING
#endif

//-----------------------------------------------------------------------------
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FCT_ACC_CUDA_ONLY or ALPAKA_FCT_ACC.
//!
//! Usage:
//! ALPAKA_FCT_ACC int add(int a, int b);
//-----------------------------------------------------------------------------
#if defined(ALPAKA_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #define ALPAKA_FCT_ACC_CUDA_ONLY __device__ __forceinline__
    #define ALPAKA_FCT_ACC_NO_CUDA __host__ __forceinline__
    #define ALPAKA_FCT_ACC \
        ALPAKA_NO_HOST_ACC_WARNING \
        __device__ __host__ __forceinline__
    #define ALPAKA_FCT_HOST_ACC \
        ALPAKA_NO_HOST_ACC_WARNING \
        __device__ __host__ __forceinline__
    #define ALPAKA_FCT_HOST __host__ __forceinline__
#else
    //#define ALPAKA_FCT_ACC_CUDA_ONLY inline
    #define ALPAKA_FCT_ACC_NO_CUDA inline
    #define ALPAKA_FCT_ACC inline
    #define ALPAKA_FCT_HOST_ACC inline
    #define ALPAKA_FCT_HOST inline
#endif

//-----------------------------------------------------------------------------
//! Suggests unrolling of the following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Unrolling in non CUDA code?
//-----------------------------------------------------------------------------
#ifdef __CUDA_ARCH__
    #if BOOST_COMP_MSVC
        #define ALPAKA_UNROLL __pragma(unroll)
    #else
        #define ALPAKA_UNROLL _Pragma("unroll")
    #endif
#else
    #define ALPAKA_UNROLL
#endif

//-----------------------------------------------------------------------------
// MSVC 2013 does not support noexcept
//-----------------------------------------------------------------------------
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
    #define noexcept
#endif

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Defines the default index type.
    //-----------------------------------------------------------------------------
    // NOTE: Setting the value type to std::size_t leads to invalid data on CUDA devices (at least with VC12).
    using UInt = std::uint32_t;

    //-----------------------------------------------------------------------------
    //! Defines implementation details that should not be used directly by the user.
    //-----------------------------------------------------------------------------
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! Rounds to the next higher power of two (if not already power of two).
        // Adapted from llvm/ADT/SmallPtrSet.h
        //-----------------------------------------------------------------------------
        template<
            unsigned N>
        struct RoundUpToPowerOfTwo;

        template<
            unsigned N,
            bool isPowerTwo>
        struct RoundUpToPowerOfTwoHelper
        {
            enum
            {
                value = N
            };
        };
        template<
            unsigned N>
        struct RoundUpToPowerOfTwoHelper<
            N,
            false>
        {
            enum
            {
                // We could just use NextVal = N+1, but this converges faster.  N|(N-1) sets
                // the right-most zero bits to one all at once, e.g. 0b0011000 -> 0b0011111.
                value = RoundUpToPowerOfTwo<(N | (N - 1)) + 1>::value
            };
        };
        template<
            unsigned N>
        struct RoundUpToPowerOfTwo
        {
            enum
            {
                value = RoundUpToPowerOfTwoHelper<N, (N&(N - 1)) == 0>::value
            };
        };

        //-----------------------------------------------------------------------------
        //! Calculates the optimal alignment for data of the given size.
        //-----------------------------------------------------------------------------
        template<
            std::size_t TuiSizeBytes>
        struct OptimalAlignment
        {
            // We have to use a enum here because VC14 says: "expected constant expression" when using "static const std::size_t".
            enum
            {
                // GCC does not support alignments larger then 128: "warning: requested alignment 256 is larger than 128[-Wattributes]".
                value = (TuiSizeBytes > 64) ? 128 : RoundUpToPowerOfTwo<TuiSizeBytes>::value
            };
        };
    }
}

#define ALPAKA_OPTIMAL_ALIGNMENT(byte)\
        ((byte)==1?1:\
        ((byte)<=2?2:\
        ((byte)<=4?4:\
        ((byte)<=8?8:\
        ((byte)<=16?16:\
        ((byte)<=32?32:\
        ((byte)<=64?64:128\
        )))))))

// Newer GCC versions >= 4.9 do not support constant expressions as parameters: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58109
#if BOOST_COMP_GNUC || BOOST_COMP_INTEL
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
    #define ALPAKA_ALIGN(TYPE, NAME) alignas(alpaka::detail::OptimalAlignment<sizeof(typename std::remove_cv<TYPE>::type)>::value) TYPE NAME
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
