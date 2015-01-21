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

#include <string>   // std::string

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The traits.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The accelerator traits.
        //-----------------------------------------------------------------------------
        namespace acc
        {
            //#############################################################################
            //! The accelerator type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetAcc;

            //#############################################################################
            //! The accelerator name trait.
            //!
            //! The default implementation returns the mangled class name.
            //#############################################################################
            template<
                typename TAcc,
                typename TSfinae = void>
            struct GetAccName
            {
                ALPAKA_FCT_HOST static std::string getAccName()
                {
                    return typeid(TAcc).name();
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    //! The accelerator trait accessors.
    //-----------------------------------------------------------------------------
    namespace acc
    {
        //#############################################################################
        //! The accelerator type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using GetAccT = typename traits::acc::GetAcc<T>::type;

        //-----------------------------------------------------------------------------
        //! Writes the accelerator name to the given stream.
        //!
        //! \tparam TAcc The accelerator type to write the name of.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc>
        ALPAKA_FCT_HOST std::string getAccName()
        {
            return traits::acc::GetAccName<TAcc>::getAccName();
        }
    }
}
