/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

// Force the usage of variadic templates for fibers.
#define BOOST_FIBERS_USE_VARIADIC_FIBER

// Boost fiber:
// http://olk.github.io/libs/fiber/doc/html/index.html
// https://github.com/olk/boost-fiber
#include <boost/fiber/fiber.hpp>                    // boost::fibers::fiber
#include <boost/fiber/operations.hpp>               // boost::this_fiber
#include <boost/fiber/condition.hpp>                // boost::fibers::condition_variable
#include <boost/fiber/mutex.hpp>                    // boost::fibers::mutex
//#include <boost/fiber/barrier.hpp>                // boost::fibers::barrier
