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

#include <type_traits>  // std::enable_if, std::is_integral
#include <climits>      // CHAR_BIT
#include <cstddef>      // std::size_t

// workarounds
#include <boost/predef.h>

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
#if defined(__CUDACC__)
    #define ALPAKA_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
#else
    #define ALPAKA_NO_HOST_ACC_WARNING
#endif

//-----------------------------------------------------------------------------
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FCT_ACC or ALPAKA_FCT_HOST_ACC.
//!
//! Usage:
//! ALPAKA_FCT_HOST_ACC int add(int a, int b);
//-----------------------------------------------------------------------------
#if defined ALPAKA_CUDA_ENABLED
    #define ALPAKA_FCT_ACC __device__ __forceinline__
    #define ALPAKA_FCT_HOST_ACC \
        ALPAKA_NO_HOST_ACC_WARNING \
        __device__ __host__ __forceinline__
    #define ALPAKA_FCT_HOST __host__ inline
#else
    #define ALPAKA_FCT_ACC inline
    #define ALPAKA_FCT_HOST_ACC inline
    #define ALPAKA_FCT_HOST inline
#endif

namespace alpaka
{
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! Rounds to the next higher power of two (if not already power of two).
        // Adapted from llvm/ADT/SmallPtrSet.h
        //-----------------------------------------------------------------------------
        template<unsigned N>
        struct RoundUpToPowerOfTwo;

        template<unsigned N, bool isPowerTwo>
        struct RoundUpToPowerOfTwoH
        {
            enum
            {
                value = N
            };
        };
        template<unsigned N>
        struct RoundUpToPowerOfTwoH<N, false>
        {
            enum
            {
                // We could just use NextVal = N+1, but this converges faster.  N|(N-1) sets
                // the right-most zero bits to one all at once, e.g. 0b0011000 -> 0b0011111.
                value = RoundUpToPowerOfTwo<(N | (N - 1)) + 1>::value
            };
        };
        template<unsigned N>
        struct RoundUpToPowerOfTwo
        {
            enum
            {
                value = RoundUpToPowerOfTwoH<N, (N&(N - 1)) == 0>::value
            };
        };

        //-----------------------------------------------------------------------------
        //! Calculates the optimal alignment for data of the given size.
        //-----------------------------------------------------------------------------
        template<std::size_t TuiSizeBytes>
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

// Older versions of GCC < 4.8 do not support alignas.
// But even newer GCC versions do not support constant expressions as parameters: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58109
#if BOOST_COMP_GNUC
    /*#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(4, 8, 0)
        #define ALPAKA_OPTIMAL_ALIGNMENT_GCC_4_8(byte) \
			        alignas(((byte)==1?1:     \
			        ((byte)<=2?2:             \
			        ((byte)<=4?4:             \
			        ((byte)<=8?8:             \
			        ((byte)<=16?16:           \
			        ((byte)<=32?32:           \
			        ((byte)<=64?64:128        \
			        ))))))))
        //-----------------------------------------------------------------------------
        //! Aligns the data optimally.
        //! You must align all arrays and structs which can used on accelerators.
        //-----------------------------------------------------------------------------
        #define ALPAKA_ALIGN(type, name) ALPAKA_OPTIMAL_ALIGNMENT_GCC_4_8(sizeof(type)) type name
        //-----------------------------------------------------------------------------
        //! Aligns the data at 8 bytes.
        //! You must align all arrays and structs which can used on accelerators.
        //-----------------------------------------------------------------------------
        #define ALPAKA_ALIGN_8(type, name) alignas(8) type name
    #else*/
        #define ALPAKA_OPTIMAL_ALIGNMENT_GCC(byte) \
			                __attribute__((aligned(((byte)==1?1:     \
			                ((byte)<=2?2:             \
			                ((byte)<=4?4:             \
			                ((byte)<=8?8:             \
			                ((byte)<=16?16:           \
			                ((byte)<=32?32:           \
			                ((byte)<=64?64:128        \
			                ))))))))))
        //-----------------------------------------------------------------------------
        //! Aligns the data optimally.
        //! You must align all arrays and structs which can used on accelerators.
        //-----------------------------------------------------------------------------
        #define ALPAKA_ALIGN(type, name) ALPAKA_OPTIMAL_ALIGNMENT_GCC(sizeof(type)) type name
        //-----------------------------------------------------------------------------
        //! Aligns the data at 8 bytes.
        //! You must align all arrays and structs which can used on accelerators.
        //-----------------------------------------------------------------------------
        #define ALPAKA_ALIGN_8(type, name) __attribute__((aligned(8))) type name
    //#endif
#else
    //-----------------------------------------------------------------------------
    //! Aligns the data optimally.
    //! You must align all arrays and structs which can used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN(type, name) alignas(alpaka::detail::OptimalAlignment<sizeof(type)>::value) type name
    //-----------------------------------------------------------------------------
    //! Aligns the data at 8 bytes.
    //! You must align all arrays and structs which can used on accelerators.
    //-----------------------------------------------------------------------------
    #define ALPAKA_ALIGN_8(type, name) alignas(8) type name
#endif
