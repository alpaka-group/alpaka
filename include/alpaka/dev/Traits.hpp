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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The device specifics.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The device traits.
        //-----------------------------------------------------------------------------
        namespace traits
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
            //! The device get trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetDev;

            //#############################################################################
            //! The device name get trait.
            //#############################################################################
            template<
                typename TDev,
                typename TSfinae = void>
            struct GetName;

            //#############################################################################
            //! The device memory size get trait.
            //#############################################################################
            template<
                typename TDev,
                typename TSfinae = void>
            struct GetMemBytes;

            //#############################################################################
            //! The device free memory size get trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetFreeMemBytes;

            //#############################################################################
            //! The device reset trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct Reset;
        }

        //#############################################################################
        //! The device type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using Dev = typename traits::DevType<T>::type;

        //#############################################################################
        //! The device manager type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using DevMan = typename traits::DevManType<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The device this object is bound to.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FN_HOST auto getDev(
            T const & t)
        -> decltype(traits::GetDev<T>::getDev(t))
        {
            return traits::GetDev<
                T>
            ::getDev(
                t);
        }

        //-----------------------------------------------------------------------------
        //! \return All the devices available on this accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TDevMan>
        ALPAKA_FN_HOST auto getDevs()
        -> std::vector<Dev<TDevMan>>
        {
            std::vector<Dev<TDevMan>> devs;

            std::size_t const devCount(TDevMan::getDevCount());
            for(std::size_t devIdx(0); devIdx < devCount; ++devIdx)
            {
                devs.push_back(TDevMan::getDevByIdx(devIdx));
            }

            return devs;
        }

        //-----------------------------------------------------------------------------
        //! \return The device name.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FN_HOST auto getName(
            TDev const & dev)
        -> std::string
        {
            return traits::GetName<
                TDev>
            ::getName(
                dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The memory on the device in Bytes.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FN_HOST auto getMemBytes(
            TDev const & dev)
        -> std::size_t
        {
            return traits::GetMemBytes<
                TDev>
            ::getMemBytes(
                dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The free memory on the device in Bytes.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FN_HOST auto getFreeMemBytes(
            TDev const & dev)
        -> std::size_t
        {
            return traits::GetFreeMemBytes<
                TDev>
            ::getFreeMemBytes(
                dev);
        }

        //-----------------------------------------------------------------------------
        //! Resets the device.
        //! What this method does is dependent of the accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FN_HOST auto reset(
            TDev const & dev)
        -> void
        {
            traits::Reset<
                TDev>
            ::reset(
                dev);
        }
    }
}
