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

#include <cuda.h>                                   // cudaGetErrorString

#include <boost/preprocessor/stringize.hpp>         // BOOST_PP_STRINGIZE

#include <iostream>                                 // std::cerr
#include <string>                                   // std::string
#include <stdexcept>                                // std::runtime_error

#if (!defined(CUDA_VERSION) || (CUDA_VERSION < 7000))
    #error "CUDA version 7.0 or greater required!"
#endif

//-----------------------------------------------------------------------------
 //! Error checking log only.
//-----------------------------------------------------------------------------
#define ALPAKA_CUDA_CHECK_MSG(cmd, msg)\
    {\
        cudaError_t const error(cmd);\
        /* Even if we get the error directly from the command, we have to reset the global error state by getting it.*/ \
        cudaGetLastError();\
        if(error != cudaSuccess)\
        {\
            std::string const sError(__FILE__ "(" BOOST_PP_STRINGIZE(__LINE__) ") '" #cmd "' returned error: '" + std::string(cudaGetErrorString(error)) + "' (possibly from a previos CUDA call)");\
            std::cerr << sError << std::endl;\
            ALPAKA_DEBUG_BREAK;\
        }\
    }

namespace alpaka
{
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! \return True if the first argument equals one of the following arguments.
        //-----------------------------------------------------------------------------
        template<
            typename T0>
        bool firstArgumentEqualsOneOfFollowing(
            T0 &&)
        {
            return false;
        }
        //-----------------------------------------------------------------------------
        //! \return True if the first argument equals one of the following arguments.
        //-----------------------------------------------------------------------------
        template<
            typename T0,
            typename T1,
            typename... TArgs>
        bool firstArgumentEqualsOneOfFollowing(
            T0 && val0,
            T1 && val1,
            TArgs && ... args)
        {
            return (val0 == val1) 
                || firstArgumentEqualsOneOfFollowing(std::forward<T0>(val0), std::forward<TArgs>(args)...);
        }
    }
}
#if BOOST_COMP_MSVC
    //-----------------------------------------------------------------------------
    //! Error checking with log and exception, ignoring specific error values
    //-----------------------------------------------------------------------------
    #define ALPAKA_CUDA_CHECK_MSG_EXCP_IGNORE(cmd, ...)\
        {\
            cudaError_t const error(cmd);\
            /* Even if we get the error directly from the command, we have to reset the global error state by getting it.*/ \
            cudaGetLastError();\
            if(error != cudaSuccess)\
            {\
                if(!alpaka::detail::firstArgumentEqualsOneOfFollowing(error, __VA_ARGS__))\
                {\
                    std::string const sError(__FILE__ "(" BOOST_PP_STRINGIZE(__LINE__) ") '" #cmd "' returned error: '" + std::string(cudaGetErrorString(error)) + "' (possibly from a previos CUDA call)");\
                    std::cerr << sError << std::endl;\
                    ALPAKA_DEBUG_BREAK;\
                    throw std::runtime_error(sError);\
                }\
            }\
        }
#else
    //-----------------------------------------------------------------------------
    //! Error checking with log and exception, ignoring specific error values
    //-----------------------------------------------------------------------------
    #define ALPAKA_CUDA_CHECK_MSG_EXCP_IGNORE(cmd, ...)\
        {\
            cudaError_t const error(cmd);\
            /* Even if we get the error directly from the command, we have to reset the global error state by getting it.*/ \
            cudaGetLastError();\
            if(error != cudaSuccess)\
            {\
                if(!alpaka::detail::firstArgumentEqualsOneOfFollowing(error, ##__VA_ARGS__))\
                {\
                    std::string const sError(__FILE__ "(" BOOST_PP_STRINGIZE(__LINE__) ") '" #cmd "' returned error: '" + std::string(cudaGetErrorString(error)) + "' (possibly from a previos CUDA call)");\
                    std::cerr << sError << std::endl;\
                    ALPAKA_DEBUG_BREAK;\
                    throw std::runtime_error(sError);\
                }\
            }\
        }
#endif

//-----------------------------------------------------------------------------
//! Error checking with log and exception.
//-----------------------------------------------------------------------------
#define ALPAKA_CUDA_CHECK_MSG_EXCP(cmd)\
    ALPAKA_CUDA_CHECK_MSG_EXCP_IGNORE(cmd)

//-----------------------------------------------------------------------------
//! The default error checking.
//-----------------------------------------------------------------------------
#define ALPAKA_CUDA_CHECK(cmd)\
    ALPAKA_CUDA_CHECK_MSG_EXCP(cmd)

