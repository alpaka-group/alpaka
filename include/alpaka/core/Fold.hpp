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

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename F,
        typename T>
    T foldr(
        F const & f,
        T const & t)
    {
        boost::ignore_unused(f);
        return t;
    }
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename F,
        typename T0,
        typename T1,
        typename... Ts>
    auto foldr(
        F const & f,
        T0 const & t0,
        T1 const & t1,
        Ts const & ... ts)
    -> decltype(f(t0, foldr(f, t1, ts...)))
    {
        return f(t0, foldr(f, t1, ts...));
    }
}