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

#include <alpaka/size/Traits.hpp>           // Size

#include <alpaka/vec/Vec.hpp>               // Vec<N>
#include <alpaka/core/Positioning.hpp>      // origin::Grid/Blocks, unit::Blocks, unit::Threads
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC

#include <type_traits>                      // std::enable_if, std::is_base_of, std::is_same, std::decay
#include <utility>                          // std::forward

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The work division traits specifics.
    //-----------------------------------------------------------------------------
    namespace workdiv
    {
        //-----------------------------------------------------------------------------
        //! The work division traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The work div trait.
            //#############################################################################
            template<
                typename TWorkDiv,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetWorkDiv;
        }

        //-----------------------------------------------------------------------------
        //! Get the extents requested.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto getWorkDiv(
            TWorkDiv const & workDiv)
        -> Vec<dim::Dim<TWorkDiv>, size::Size<TWorkDiv>>
        {
            return
                traits::GetWorkDiv<
                    TWorkDiv,
                    TOrigin,
                    TUnit>
                ::getWorkDiv(
                    workDiv);
        }

        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers block thread extents trait specialization for classes with WorkDivBase member type.
            //#############################################################################
            template<
                typename TWorkDiv,
                typename TOrigin,
                typename TUnit>
            struct GetWorkDiv<
                TWorkDiv,
                TOrigin,
                TUnit,
                typename std::enable_if<
                    std::is_base_of<typename TWorkDiv::WorkDivBase, typename std::decay<TWorkDiv>::type>::value
                    && (!std::is_same<typename TWorkDiv::WorkDivBase, typename std::decay<TWorkDiv>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, size::Size<TWorkDiv>>
                {
                    // Delegate the call to the base class.
                    return
                        workdiv::getWorkDiv<
                            TOrigin,
                            TUnit>(
                                static_cast<typename TWorkDiv::WorkDivBase const &>(workDiv));
                }
            };

            //#############################################################################
            //! The work div grid thread extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                -> Vec<dim::Dim<typename TWorkDiv::WorkDivBase>, size::Size<TWorkDiv>>
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                        * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv);
                }
            };
        }
    }
}
