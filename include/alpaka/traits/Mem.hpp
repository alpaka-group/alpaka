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

#include <alpaka/traits/mem/Space.hpp>  // SpaceT
#include <alpaka/traits/mem/Buf.hpp>    // Alloc
#include <alpaka/traits/mem/View.hpp>   // ViewType

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The memory traits.
        //-----------------------------------------------------------------------------
        namespace mem
        {
            //#############################################################################
            //! The memory buffer base type trait.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim,
                typename TSfinae = void>
            struct BufType;
        }
    }

    //-----------------------------------------------------------------------------
    //! The memory trait accessors.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        //#############################################################################
        //! The memory buffer base type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TElem,
            typename TDim,
            typename TSpace>
        using BufT = typename traits::mem::BufType<TElem, TDim, SpaceT<TSpace>>::type;
    }
}