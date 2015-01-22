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

#include <alpaka/core/DevProps.hpp> // DevProps

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The device traits.
        //-----------------------------------------------------------------------------
        namespace dev
        {
            //#############################################################################
            //! The device type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetDev;

            //#############################################################################
            //! The device manager type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetDevMan;

            //#############################################################################
            //! The device properties get trait.
            //#############################################################################
            template<
                typename TDev,
                typename TSfinae = void>
            struct GetDevProps;
        }
    }

    //-----------------------------------------------------------------------------
    //! The device trait accessors.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        //#############################################################################
        //! The device type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using GetDevT = typename traits::dev::GetDev<T>::type;

        //#############################################################################
        //! The device manager type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using GetDevManT = typename traits::dev::GetDevMan<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The device properties.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FCT_HOST DevProps getDevProps(
            TDev const & device)
        {
            return traits::dev::GetDevProps<TDev>::getDevProps(device);
        }
    }
}
