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

#include <alpaka/cuda/mem/Space.hpp>// SpaceCuda
#include <alpaka/cuda/mem/Copy.hpp> // Copy
#include <alpaka/cuda/mem/Buf.hpp>  // BufCuda

#include <alpaka/core/mem/View.hpp> // View

#include <alpaka/traits/Mem.hpp>    // traits::mem::BufT

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The BufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct BufType<
                TElem, TDim, alpaka::mem::SpaceCuda>
            {
                using type = cuda::detail::BufCuda<TElem, TDim>;
            };

            //#############################################################################
            //! The BufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct ViewType<
                cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = alpaka::mem::detail::View<cuda::detail::BufCuda<TElem, TDim>>;
            };
        }
    }
}
