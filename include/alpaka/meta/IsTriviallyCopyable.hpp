/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>

#include <type_traits>

#if BOOST_LANG_CUDA
#include <nvfunctional>
#endif

namespace alpaka
{
    namespace meta
    {
        template<
            class T>
        struct IsTriviallyCopyable
            : std::integral_constant<bool,
#if BOOST_LIB_STD_GNU
                __has_trivial_copy(T)
#else
                std::is_trivially_copyable<T>::value
#endif
#if BOOST_LANG_CUDA && !BOOST_COMP_CLANG_CUDA
                || __nv_is_extended_device_lambda_closure_type(T)
                || __nv_is_extended_host_device_lambda_closure_type(T)
#endif
            >
        {
        };

#if BOOST_LANG_CUDA
        template<
            class T>
        struct IsTriviallyCopyable<nvstd::function<T>>
            : std::integral_constant<bool, true>
        {
        };
#endif
    }
}
