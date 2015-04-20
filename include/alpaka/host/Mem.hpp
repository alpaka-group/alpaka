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

#include <alpaka/host/mem/Space.hpp>// SpaceHost
#include <alpaka/host/mem/Copy.hpp> // Copy
#include <alpaka/host/mem/Buf.hpp>  // BufHost

#include <alpaka/core/mem/View.hpp> // View

#include <alpaka/traits/Mem.hpp>    // traits::mem::BufType

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The BufHost memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim>
            struct BufType<
                TDev, TElem, TDim, alpaka::mem::SpaceHost>
            {
                using type = host::detail::BufHost<TDev, TElem, TDim>;
            };

            //#############################################################################
            //! The BufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim>
            struct ViewType<
                host::detail::BufHost<TDev, TElem, TDim>>
            {
                using type = alpaka::mem::detail::View<host::detail::BufHost<TDev, TElem, TDim>>;
            };
        }
    }
}
