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
#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

#include <type_traits>                  // std::enable_if, std::is_integral, std::is_unsigned

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
            //! The dimension getter type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct DimType;
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
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        template<
            typename T>
        struct DimT
        {
            using type = typename traits::dim::DimType<T>::type;
            static const UInt value = type::value;
        };
#else
        template<
            typename T>
        using DimT = typename traits::dim::DimType<T>::type;
#endif


        //-----------------------------------------------------------------------------
        //! \return The dimension.
        //-----------------------------------------------------------------------------
        template<
            typename T>
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        ALPAKA_FCT_HOST_ACC auto getDim()
#else
        ALPAKA_FCT_HOST_ACC auto constexpr getDim()
#endif
        -> UInt
        {
            return DimT<T>::value;
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The unsigned integral dimension getter trait specialization.
            //#############################################################################
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                using type = alpaka::dim::Dim1;
            };
        }
    }
}
