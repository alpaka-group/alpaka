/**
* \file
* Copyright 2014-2015 Benjarint Worpitz, Rene Widera
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

#include <alpaka/math/trunc/Traits.hpp> // Trunc

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_arithmetic
#include <math_functions.hpp>           // ::trunc

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library trunc.
        //#############################################################################
        class TruncCudaBuiltIn
        {
        public:
            using TruncBase = TruncCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library trunc trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Trunc<
                TruncCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto trunc(
                    TruncCudaBuiltIn const & /*trunc*/,
                    TArg const & arg)
                -> decltype(::trunc(arg))
                {
                    return ::trunc(arg);
                }
            };
        }
    }
}
