/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Matthias Werner, Jan Stephan, René Widera, Andrea Bocci, Aurora Perego,
 * Tim Hanel, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#include <type_traits>


#if ALPAKA_LANG_HIP
// HIP defines some keywords like __forceinline__ in header files.
#    include <hip/hip_runtime.h>
#endif

//! Function declaration attributes to provide execution scope information
//!
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FN_ACC or ALPAKA_FN_HOST_ACC.
//! Deduction guides for objects usable within a kernel should be marked ALPAKA_FN_DG.
//! ALPAKA_FN_DG declare the guide for on host and acc usage.
//!
//! \code{.cpp}
//! // function
//! ALPAKA_FN_ACC std::int32_t add(std::int32_t a, std::int32_t b);
//!
//! // struct with deduction guide
//! template<typename T>
//! struct Foo
//! {
//!     T bar;
//!
//!     template<typename T_ValueType>
//!     constexpr Foo(T_ValueType const v): bar{static_cast<T>(v)}
//!     {}
//! };
//!
//! template<typename T_ValueType>
//! ALPAKA_FN_DG Foo(T_ValueType const) -> Foo<T_ValueType>;
//! \endcode
//! @{
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    define ALPAKA_FN_ACC __device__ __host__
#    define ALPAKA_FN_HOST_ACC __device__ __host__
#    define ALPAKA_FN_HOST __host__
// Avoid warning: use of CUDA/HIP target attributes on deduction guides is deprecated; they will be rejected in a
// future version of Clang [-Werror,-Wdeprecated-attributes]
#    if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(22, 1, 0)
#        define ALPAKA_FN_DG
#    else
#        define ALPAKA_FN_DG __device__ __host__
#    endif
#else
#    define ALPAKA_FN_ACC
#    define ALPAKA_FN_HOST_ACC
#    define ALPAKA_FN_HOST
#    define ALPAKA_FN_DG
#endif
//! @}

//! All functions marked with ALPAKA_FN_ACC or ALPAKA_FN_HOST_ACC that are exported to / imported from different
//! translation units have to be attributed with ALPAKA_FN_EXTERN. Note that this needs to be applied to both the
//! declaration and the definition.
//!
//! Usage:
//! ALPAKA_FN_ACC ALPAKA_FN_EXTERN auto add(std::int32_t a, std::int32_t b) -> std::int32_t;
//!
//! Warning: If this is used together with the SYCL back-end make sure that your SYCL runtime supports generic
//! address spaces. Otherwise it is forbidden to use pointers as parameter or return type for functions marked
//! with ALPAKA_FN_EXTERN.
#if ALPAKA_LANG_SYCL
/*
   This is required by the SYCL standard, section 5.10.1 "SYCL functions and member functions linkage":

   The default behavior in SYCL applications is that all the definitions and declarations of the functions and member
   functions are available to the SYCL compiler, in the same translation unit. When this is not the case, all the
   symbols that need to be exported to a SYCL library or from a C++ library to a SYCL application need to be defined
   using the macro: SYCL_EXTERNAL.
*/
#    define ALPAKA_FN_EXTERN SYCL_EXTERNAL
#else
#    define ALPAKA_FN_EXTERN
#endif

//! Disable nvcc warning:
//! 'calling a __host__ function from __host__ __device__ function.'
//! Usage:
//! ALPAKA_NO_HOST_ACC_WARNING
//! ALPAKA_FN_HOST_ACC function_declaration()
//! WARNING: Only use this method if there is no other way.
//! Most cases can be solved by #if ALPAKA_ARCH_PTX or #if ALPAKA_LANG_CUDA.
#if (ALPAKA_LANG_CUDA && !ALPAKA_COMP_CLANG_CUDA)
#    if ALPAKA_COMP_MSVC
#        define ALPAKA_NO_HOST_ACC_WARNING __pragma(hd_warning_disable)
#    else
#        define ALPAKA_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
#    endif
#else
#    define ALPAKA_NO_HOST_ACC_WARNING
#endif

//! Macro defining the inline function attribute.
//!
//! The macro should stay on the left hand side of keywords, e.g. 'static', 'constexpr', 'explicit' or the return type.
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    define ALPAKA_FN_INLINE __forceinline__
#elif ALPAKA_COMP_MSVC
// TODO: With C++20 [[msvc::forceinline]] can be used.
#    define ALPAKA_FN_INLINE __forceinline
#else
// For gcc, clang, and clang-based compilers like Intel icpx
#    define ALPAKA_FN_INLINE [[gnu::always_inline]] inline
#endif

/** Inline lambda function and add additional attributes.
 *
 * Places the always_inline attribute where the compiler accepts it in a lambda
 * declaration.
 * Pass lambda specifiers such as mutable as arguments.
 *
 * @code{.cpp}
 * auto foo = []() ALPAKA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr){ return 1;}
 * @endcode
 */
#if (ALPAKA_COMP_CLANG && !ALPAKA_COMP_NVCC) || ALPAKA_COMP_ICPX
#    define ALPAKA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __attribute__((always_inline)) __VA_ARGS__
#elif ALPAKA_COMP_GNUC || ALPAKA_COMP_NVCC
#    define ALPAKA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__ __attribute__((always_inline))
#else
#    define ALPAKA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__
#    warning ALPAKA_LAMBDA_INLINE_WITH_SPECIFIERS not defined for this compiler
#endif

//! Gives strong indication to the compiler to inline the attributed lambda.
#define ALPAKA_LAMBDA_INLINE ALPAKA_LAMBDA_INLINE_WITH_SPECIFIERS()

//! This macro defines a variable lying in global accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_ACC_MEM_GLOBAL alpaka::DevGlobal<TAcc, int> variable;
//!
//! Those variables behave like ordinary variables when used in file-scope,
//! but inside kernels the get() method must be used to access the variable.
//! They are declared inline to resolve to a single instance across multiple
//! translation units.
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
//!
//! \attention It is not allowed to initialize the variable together with the declaration.
//!            To initialize the variable alpaka::memcpy must be used.
//! \code{.cpp}
//! ALPAKA_STATIC_ACC_MEM_GLOBAL alpaka::DevGlobal<TAcc, int> foo;
//!
//! struct DeviceMemoryKernel
//! {
//!    ALPAKA_NO_HOST_ACC_WARNING
//!    template<typename TAcc>
//!    ALPAKA_FN_ACC void operator()(TAcc const& acc) const
//!    {
//!      auto a = foo<TAcc>.get();
//!    }
//!  }
//!
//! void initFoo() {
//!     auto extent = alpaka::Vec<alpaka::DimInt<1u>, size_t>{1};
//!     int initialValue = 42;
//!     alpaka::ViewPlainPtr<DevHost, int, alpaka::DimInt<1u>, size_t> bufHost(&initialValue, devHost, extent);
//!     alpaka::memcpy(queue, foo<Acc>, bufHost, extent);
//! }
//! \endcode
#if (                                                                                                                 \
    (ALPAKA_LANG_CUDA && ALPAKA_COMP_CLANG_CUDA) || (ALPAKA_LANG_CUDA && ALPAKA_COMP_NVCC && ALPAKA_ARCH_PTX)         \
    || ALPAKA_LANG_HIP)
#    if defined(__CUDACC_RDC__) || defined(__CLANG_RDC__)
#        define ALPAKA_STATIC_ACC_MEM_GLOBAL                                                                          \
            template<typename TAcc>                                                                                   \
            __device__ inline
#    else
#        define ALPAKA_STATIC_ACC_MEM_GLOBAL                                                                          \
            template<typename TAcc>                                                                                   \
            __device__ static
#    endif
#else
#    define ALPAKA_STATIC_ACC_MEM_GLOBAL                                                                              \
        template<typename TAcc>                                                                                       \
        inline
#endif

/** Perfectly forward an instance as argument. */
#define ALPAKA_FORWARD(instance) std::forward<decltype(instance)>(instance)

/** Get the type of instance
 *
 * References will be removed which is often required because traits are mostly defined for the type only.
 */
#define ALPAKA_TYPEOF(...) std::decay_t<decltype(__VA_ARGS__)>
