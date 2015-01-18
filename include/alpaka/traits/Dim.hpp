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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>          // alpaka::Vec
#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

#include <cstdint>                      // std::size_t

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The dimension traits.
        //-----------------------------------------------------------------------------
        namespace dim
        {
            //#############################################################################
            //! The dimension getter trait.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetDim;

            //#############################################################################
            //! The dimension to vector transformation trait.
            //#############################################################################
            template<
                typename TDim>
            struct DimToVec
            {
                using type = Vec<TDim::value>;
            };
        }
    }

    //-----------------------------------------------------------------------------
    //! The dimension trait accessors.
    //-----------------------------------------------------------------------------
    namespace dim
    {
        //#############################################################################
        //! The dimension getter trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using GetDimT = typename traits::dim::GetDim<T>::type;
        //-----------------------------------------------------------------------------
        //! \return The dimension.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST std::size_t getDim()
        {
            return GetDimT<T>::value;
        }

        //#############################################################################
        //! The dimension to vector alias template to remove the ::type.
        //#############################################################################
        template<
            typename TDim>
        using DimToVecT = typename traits::dim::DimToVec<TDim>::type;
    }
}
