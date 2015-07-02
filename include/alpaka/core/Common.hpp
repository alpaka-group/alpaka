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
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #if BOOST_COMP_MSVC
        #define ALPAKA_NO_HOST_ACC_WARNING\
            __pragma(hd_warning_disable)
    #else
        #define ALPAKA_NO_HOST_ACC_WARNING\
            _Pragma("hd_warning_disable")
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
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #define ALPAKA_FCT_ACC_CUDA_ONLY\
        ALPAKA_NO_HOST_ACC_WARNING\
        __device__ __forceinline__
    #define ALPAKA_FCT_ACC_NO_CUDA\
        ALPAKA_NO_HOST_ACC_WARNING\
        __host__ __forceinline__
    #define ALPAKA_FCT_ACC\
        ALPAKA_NO_HOST_ACC_WARNING\
        __device__ __host__ __forceinline__
    #define ALPAKA_FCT_HOST_ACC\
        ALPAKA_NO_HOST_ACC_WARNING\
        __device__ __host__ __forceinline__
    #define ALPAKA_FCT_HOST\
        ALPAKA_NO_HOST_ACC_WARNING\
        __host__ __forceinline__
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
        #define ALPAKA_UNROLL(...) __pragma(unroll##__VA_ARGS__)
    #else
        #define ALPAKA_UNROLL(...)  _Pragma("unroll"##__VA_ARGS__)
    #endif
#else
    #define ALPAKA_UNROLL(...)
#endif

namespace alpaka
{
    //#############################################################################
    //! A false_type being dependent on a ignored template parameter.
    //! This allows to use static_assert in uninstantiated template specializations without triggering.
    //#############################################################################
    template<
        typename T>
    struct dependent_false_type :
        std::false_type
    {};
}
