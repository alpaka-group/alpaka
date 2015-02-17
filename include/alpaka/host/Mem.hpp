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

#include <alpaka/host/MemSpace.hpp>         // MemSpaceHost
#include <alpaka/host/mem/MemCopy.hpp>      // MemCopy
#include <alpaka/host/mem/MemBufBase.hpp>   // MemBufBaseHost

#include <alpaka/core/mem/MemBufView.hpp>   // MemBufView

#include <alpaka/traits/Mem.hpp>            // traits::mem::MemBufBaseType

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The MemBufBaseHost memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct MemBufBaseType<
                TElem, TDim, alpaka::mem::MemSpaceHost>
            {
                using type = host::detail::MemBufBaseHost<TElem, TDim>;
            };

            //#############################################################################
            //! The MemBufBaseCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct MemBufViewType<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                using type = alpaka::mem::detail::MemBufView<host::detail::MemBufBaseHost<TElem, TDim>>;
            };
        }
    }
}
