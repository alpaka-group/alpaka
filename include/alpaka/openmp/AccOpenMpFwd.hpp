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

#include <alpaka/traits/Acc.hpp>        // GetAccName
#include <alpaka/traits/Memory.hpp>     // GetMemSpace

#include <alpaka/host/MemorySpace.hpp>  // MemSpaceHost

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The OpenMP accelerator.
    //-----------------------------------------------------------------------------
    namespace openmp
    {
        namespace detail
        {
            // forward declarations
            class AccOpenMp;
        }
    }
    using AccOpenMp = openmp::detail::AccOpenMp;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                openmp::detail::AccOpenMp>
            {
                static std::string getAccName()
                {
                    return "AccOpenMp";
                }
            };
        }

        namespace memory
        {
            //#############################################################################
            //! The OpenMP accelerator memory space trait specialization.
            //#############################################################################
            template<>
            struct GetMemSpace<
                openmp::detail::AccOpenMp>
            {
                using type = alpaka::memory::MemSpaceHost;
            };
        }
    }
}
