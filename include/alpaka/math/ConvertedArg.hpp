/* Copyright 2021 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/Complex.hpp>

#include <complex>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            namespace detail
            {
                //! Wrapper for argument types in StdLib implementations of math functions
                //!
                //! Defines the type of the argument after conversion (if conversion is necessary).
                //! Default implementation passes through the original type.
                //!
                //! \tparam TArg original argument type
                template<typename TArg>
                struct ConvertedArg
                {
                    using type = TArg;
                };

                //! Specialization for alpaka::Complex, convert to std::complex
                template<typename T>
                struct ConvertedArg<Complex<T>>
                {
                    using type = std::complex<T>;
                };
            } // namespace detail

            //! Wrapper for argument types in StdLib implementations of math functions
            //!
            //! Defines the type of the argument after conversion (if conversion is necessary).
            //! Helps provide conversion and avoid duplication of implementations.
            //!
            //! \tparam TArg original argument type
            template<typename TArg>
            using ConvertedArg = typename detail::ConvertedArg<TArg>::type;

        } // namespace traits
    } // namespace math
} // namespace alpaka
