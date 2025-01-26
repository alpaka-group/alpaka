/* Copyright 2025 Jan Stephan, Jeffrey Kelling, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

//! Before CUDA 11.5 nvcc is unable to correctly identify return statements in 'if constexpr' branches. It will issue
//! a false warning about a missing return statement unless it is told that the following code section is unreachable.
//!
//! \param x A dummy value for the expected return type of the calling function.
#if(BOOST_COMP_NVCC && BOOST_ARCH_PTX)
#    define ALPAKA_UNREACHABLE(...) __builtin_unreachable()
#elif BOOST_COMP_MSVC
#    define ALPAKA_UNREACHABLE(...) __assume(false)
#elif BOOST_COMP_GNUC || BOOST_COMP_CLANG
#    define ALPAKA_UNREACHABLE(...) __builtin_unreachable()
#else
#    define ALPAKA_UNREACHABLE(...)
#endif
