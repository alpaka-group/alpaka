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

#include <boost/current_function.hpp>
#include <boost/predef.h>                   // workarounds
/*#include <boost/preprocessor/stringize.hpp> // BOOST_PP_STRINGIZE

#include <stdexcept>                        // std::runtime_error*/
#include <string>                           // std::string
#include <iostream>                         // std::cout

//-----------------------------------------------------------------------------
//! The no debug level.
//-----------------------------------------------------------------------------
#define ALPAKA_DEBUG_DISABLED 0
//-----------------------------------------------------------------------------
//! The minimal debug level.
//-----------------------------------------------------------------------------
#define ALPAKA_DEBUG_MINIMAL 1
//-----------------------------------------------------------------------------
//! The full debug level.
//-----------------------------------------------------------------------------
#define ALPAKA_DEBUG_FULL 2

#ifndef ALPAKA_DEBUG
    //-----------------------------------------------------------------------------
    //! Set the minimum log level if it is not defined.
    //-----------------------------------------------------------------------------
    #define ALPAKA_DEBUG ALPAKA_DEBUG_DISABLED
#endif

//-----------------------------------------------------------------------------
// Boost does not define BOOST_CURRENT_FUNCTION for MSVC correctly.
//-----------------------------------------------------------------------------
#if BOOST_COMP_MSVC
    #undef BOOST_CURRENT_FUNCTION
    #define BOOST_CURRENT_FUNCTION __FUNCTION__
#endif

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //! Scope logger.
            //#############################################################################
            class ScopeLogStdOut final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ScopeLogStdOut(
                    std::string const & sScope) :
                        m_sScope(sScope)
                {
                    std::cout << "[+] " << m_sScope << std::endl;
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ScopeLogStdOut(ScopeLogStdOut const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ScopeLogStdOut(ScopeLogStdOut &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ScopeLogStdOut const &) -> ScopeLogStdOut & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ScopeLogStdOut &&) -> ScopeLogStdOut & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ~ScopeLogStdOut()
                {
                    std::cout << "[-] " << m_sScope << std::endl;
                }

            private:
                std::string const m_sScope;
            };
        }
    }
}

//-----------------------------------------------------------------------------
// Define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE.
//-----------------------------------------------------------------------------
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE\
        ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(BOOST_CURRENT_FUNCTION)
#else
    #define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE
#endif

//-----------------------------------------------------------------------------
// Define ALPAKA_DEBUG_FULL_LOG_SCOPE.
//-----------------------------------------------------------------------------
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    #define ALPAKA_DEBUG_FULL_LOG_SCOPE\
        ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(BOOST_CURRENT_FUNCTION)
#else
    #define ALPAKA_DEBUG_FULL_LOG_SCOPE
#endif

//-----------------------------------------------------------------------------
// Define ALPAKA_DEBUG_BREAK.
//-----------------------------------------------------------------------------
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #if BOOST_COMP_GNUC
        #define ALPAKA_DEBUG_BREAK ::__builtin_trap()
    #elif BOOST_COMP_INTEL
        #define ALPAKA_DEBUG_BREAK ::__debugbreak()
    #elif BOOST_COMP_MSVC
        #define ALPAKA_DEBUG_BREAK ::__debugbreak()
    #else
        #define ALPAKA_DEBUG_BREAK
        //#error debug-break for current compiler not implemented!
    #endif
#else
    #define ALPAKA_DEBUG_BREAK
#endif

//-----------------------------------------------------------------------------
//! Error checking with log and exception.
//-----------------------------------------------------------------------------
/*#define ALPAKA_ASSERT_MSG_EXCP1(assertionExpression)\
    {\
        bool const error(assertionExpression);\
        if(error != cudaSuccess)\
        {\
            std::string const sError(__FILE__ "(" BOOST_PP_STRINGIZE(__LINE__) ") '" #assertionExpression "' assertion failed!");\
            std::cerr << sError << std::endl;\
            ALPAKA_DEBUG_BREAK;\
            throw std::runtime_error(sError);\
        }\
    }
//-----------------------------------------------------------------------------
//! Error checking with log and exception.
//-----------------------------------------------------------------------------
#define ALPAKA_ASSERT_MSG_EXCP2(assertionExpression, sErrorMessage)\
    {\
        bool const error(assertionExpression);\
        if(error != cudaSuccess)\
        {\
            std::string const sError(__FILE__ "(" BOOST_PP_STRINGIZE(__LINE__) ") '" #assertionExpression "' assertion failed: '" + std::string(sErrorMessage) + "'!");\
            std::cerr << sError << std::endl;\
            ALPAKA_DEBUG_BREAK;\
            throw std::runtime_error(sError);\
        }\
    }

#define ALPAKA_ASSERT_MSG_EXCP_GET(_1,_2,NAME,...) NAME
#define ALPAKA_ASSERT_MSG_EXCP(...)\
    ALPAKA_ASSERT_MSG_EXCP_GET(__VA_ARGS__, ALPAKA_ASSERT_MSG_EXCP2, ALPAKA_ASSERT_MSG_EXCP1)(__VA_ARGS__)*/
