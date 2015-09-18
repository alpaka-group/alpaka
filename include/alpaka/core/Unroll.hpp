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

#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC

#if (!defined(__CUDA_ARCH__))
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

//-----------------------------------------------------------------------------
//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
//-----------------------------------------------------------------------------
#ifdef __CUDA_ARCH__
    #if BOOST_COMP_MSVC
        #define ALPAKA_UNROLL(...) __pragma(unroll __VA_ARGS__)
    #else
        #define ALPAKA_UNROLL_STRINGIFY(x) #x
        #define ALPAKA_UNROLL(...)  _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))
    #endif
#else
    #if BOOST_COMP_INTEL || BOOST_COMP_IBM || BOOST_COMP_SUNPRO || BOOST_COMP_HPACC
        #define ALPAKA_UNROLL_STRINGIFY(x) #x
        #define ALPAKA_UNROLL(...)  _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
    #elif BOOST_COMP_PGI
        #define ALPAKA_UNROLL(...)  _Pragma("unroll")
    #else
        #define ALPAKA_UNROLL(...)
    #endif
#endif

/*namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //! Loop unroller.
            //#############################################################################
            template<
                intmax_t Tbegin,
                intmax_t Tend,
                intmax_t Tstep>
            struct Unroll;
            //#############################################################################
            //! Loop unroller.
            //#############################################################################
            template<
                intmax_t Tend,
                intmax_t Tstep>
            struct Unroll<
                Tend,
                Tend,
                Tstep>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TFnObj,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto unroll(
                    TFnObj const & f,
                    TArgs const & ... args)
                -> void
                {
#ifndef __CUDA_ARCH__
                    boost::ignore_unused(f);
                    boost::ignore_unused(args...);
#endif
                }
            };
            //#############################################################################
            //! Loop unroller.
            //#############################################################################
            template<
                intmax_t Tcurrent,
                intmax_t Tend,
                intmax_t Tstep
            struct Unroll
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TFnObj,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto unroll(
                    TFnObj const & f,
                    TArgs const & ... args)
                -> void
                {
                    f(idx, args...);

                    detail::Unroll<
                        Tcurrent+Tstep,
                        Tend,
                        Tstep>
                    ::unroll(
                        f,
                        args...);
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested from index zero outmost to index (dim-1) innermost.
        //!
        //! \param extents N-dimensional loop extents.
        //! \param f The function called at each iteration.
        //! \param args,... The additional arguments given to each function call.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            intmax_t Tbegin,
            intmax_t Tend,
            intmax_t Tstep,
            typename TFnObj,
            typename... TArgs,>
        ALPAKA_FN_HOST_ACC auto unroll(
            TFnObj const & f,
            TArgs const & ... args)
        -> void
        {
            detail::Unroll<
                Tbegin,
                Tend,
                Tstep>
            ::template unroll<
                0u>(
                    f,
                    args...);
        }
    }
}*/
