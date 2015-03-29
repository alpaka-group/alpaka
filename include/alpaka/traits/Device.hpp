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
            struct DevType;

            //#############################################################################
            //! The device manager type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct DevManType;

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
        using DevT = typename traits::dev::DevType<typename std::decay<T>::type>::type;

        //#############################################################################
        //! The device manager type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using DevManT = typename traits::dev::DevManType<typename std::decay<T>::type>::type;

        //-----------------------------------------------------------------------------
        //! \return The device properties.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FCT_HOST auto getDevProps(
            TDev const & device)
        -> DevProps
        {
            return traits::dev::GetDevProps<
                TDev>
            ::getDevProps(device);
        }

        //-----------------------------------------------------------------------------
        //! \return All the devices available on this accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TDevMan>
        ALPAKA_FCT_HOST auto getDevices(
            TDevMan const & devMan)
        -> std::vector<DevT<TDevMan>>
        {
            std::vector<DevT<TDevMan>> vDevices;

            std::size_t const uiDeviceCount(devMan.getDevCount());
            for(std::size_t uiDeviceIdx(0); uiDeviceIdx < uiDeviceCount; ++uiDeviceIdx)
            {
                vDevices.push_back(devMan.getDevByIdx(uiDeviceIdx));
            }

            return vDevices;
        }
    }
}
