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

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4267)  // boost/asio/detail/impl/socket_ops.ipp(1968): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
                                    // boost/asio/detail/impl/socket_ops.ipp(2172): warning C4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
    #pragma warning(disable: 4297)  // boost/coroutine/detail/symmetric_coroutine_impl.hpp(409) : warning C4297 : 'boost::coroutines::detail::symmetric_coroutine_impl<void>::yield' : function assumed not to throw an exception but does
    #pragma warning(disable: 4996)  // boost/asio/detail/impl/socket_ops.ipp(1363) : warning C4996 : 'WSASocketA' : Use WSASocketW() instead or define _WINSOCK_DEPRECATED_NO_WARNINGS to disable deprecated API warnings
    #pragma warning(disable: 4456)  // boost/fiber/condition.hpp(174): warning C4456: declaration of 'lk' hides previous local declaration
                                    // boost/fiber/condition.hpp(217): warning C4456: declaration of 'lk' hides previous local declaration
    // Boost.Fiber indirectly includes windows.h for which we need to define some things
    #define NOMINMAX
#endif

// Boost fiber:
// http://olk.github.io/libs/fiber/doc/html/index.html
// https://github.com/olk/boost-fiber
#include <boost/fiber/fiber.hpp>        // boost::fibers::fiber
#include <boost/fiber/operations.hpp>   // boost::this_fiber
#include <boost/fiber/condition.hpp>    // boost::fibers::condition_variable
#include <boost/fiber/mutex.hpp>        // boost::fibers::mutex
#include <boost/fiber/future.hpp>       // boost::fibers::future
//#include <boost/fiber/barrier.hpp>    // boost::fibers::barrier

#if BOOST_COMP_MSVC
    #undef NOMINMAX
    #pragma warning(pop)
#endif