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
#include <alpaka/traits/Mem.hpp>        // SpaceType

#include <alpaka/host/mem/Space.hpp>    // SpaceHost

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The fibers accelerator.
    //-----------------------------------------------------------------------------
    namespace fibers
    {
        namespace detail
        {
            // forward declarations
            class AccFibers;
        }
    }
    using AccFibers = fibers::detail::AccFibers;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The fibers accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                fibers::detail::AccFibers>
            {
                using type = fibers::detail::AccFibers;
            };

            //#############################################################################
            //! The fibers accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                fibers::detail::AccFibers>
            {
                static std::string getAccName()
                {
                    return "AccFibers";
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The fibers accelerator memory space trait specialization.
            //#############################################################################
            template<>
            struct SpaceType<
                fibers::detail::AccFibers>
            {
                using type = alpaka::mem::SpaceHost;
            };
        }
    }
}
