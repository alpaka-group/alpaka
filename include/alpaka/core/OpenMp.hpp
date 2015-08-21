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

#include <omp.h>

namespace alpaka
{
    namespace omp
    {
        //-----------------------------------------------------------------------------
        //! \return The device this object is bound to.
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto getMaxOmpThreads()
        -> int
        {
            // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP 2.0 runtime:
            // 'The omp_get_max_threads routine returns the value of the internal control variable, which is used to determine the number of threads that would form the new team,
            // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
            // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
            ::omp_set_num_threads(1024);
            return ::omp_get_max_threads();
        }
    }
}
