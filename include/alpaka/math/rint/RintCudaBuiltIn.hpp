/**
* \file
* Copyright 2014-2015 Benjarint Worpitz
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

#include <alpaka/math/rint/Traits.hpp> // Rint

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_arithmetic
#include <math_functions.hpp>           // ::rint

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library rint.
        //#############################################################################
        class RintCudaBuiltIn
        {
        public:
            using RintBase = RintCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library rint trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Rint<
                RintCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto rint(
                    RintCudaBuiltIn const & rint,
                    TArg const & arg)
                -> decltype(::rint(arg))
                {
                    boost::ignore_unused(rint);
                    return ::rint(arg);
                }
            };
            //#############################################################################
            //! The standard library rint trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Lrint<
                RintCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto lrint(
                    RintCudaBuiltIn const & lrint,
                    TArg const & arg)
                -> long int
                {
                    boost::ignore_unused(lrint);
                    return ::lrint(arg);
                }
            };
            //#############################################################################
            //! The standard library rint trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Llrint<
                RintCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto llrint(
                    RintCudaBuiltIn const & llrint,
                    TArg const & arg)
                -> long int
                {
                    boost::ignore_unused(llrint);
                    return ::llrint(arg);
                }
            };
        }
    }
}
