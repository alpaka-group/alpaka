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

#include <alpaka/traits/Acc.hpp>            // acc::getAccName
#include <alpaka/traits/Stream.hpp>         // stream::StreamT
#include <alpaka/traits/Extent.hpp>         // extent::getXXX
#include <alpaka/traits/WorkDiv.hpp>        // workdiv::getWorkDiv

#include <alpaka/core/WorkDivHelpers.hpp>   // workdiv::isValidWorkDiv
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The execution traits.
        //-----------------------------------------------------------------------------
        namespace exec
        {
            //#############################################################################
            //! The executor type trait.
            //#############################################################################
            template<
                typename TAcc,
                typename TSfinae = void>
            struct ExecType;
        }
    }

    //-----------------------------------------------------------------------------
    //! The executor trait accessors.
    //-----------------------------------------------------------------------------
    namespace exec
    {
        //#############################################################################
        //! The executor type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TAcc>
        using ExecT = typename traits::exec::ExecType<TAcc>::type;

        //-----------------------------------------------------------------------------
        //! \return A kernel executor.
        //-----------------------------------------------------------------------------
        template<
            typename TStream,
            typename TWorkDiv>
        ALPAKA_FCT_HOST auto create(
            TWorkDiv const & workDiv,
            TStream const & stream)
        -> ExecT<acc::AccT<TStream>>
        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << BOOST_CURRENT_FUNCTION
                << " gridBlockExtents: " << workdiv::getWorkDiv<Grid, Blocks>(workDiv)
                << ", blockThreadExtents: " << workdiv::getWorkDiv<Block, Threads>(workDiv)
                << std::endl;
#endif
            // Some basic tests.
            if(workdiv::getWorkDiv<Grid, Blocks, dim::Dim1>(workDiv)[0] == 0u)
            {
                throw std::runtime_error("The workDiv grid blocks extents is not allowed to be zero in any dimension!");
            }
            if(workdiv::getWorkDiv<Block, Threads, dim::Dim1>(workDiv)[0] == 0u)
            {
                throw std::runtime_error("The workDiv block thread extents is not allowed to be zero in any dimension!");
            }

            // This checks for the compliance with the maxima of the accelerator.
            if(!workdiv::isValidWorkDiv(dev::getDev(stream), workDiv))
            {
                throw std::runtime_error("The given work division is not supported by the " + acc::getAccName<acc::AccT<TStream>>() + " accelerator!");
            }

            return ExecT<acc::AccT<TStream>>(workDiv, stream);
        }
        //-----------------------------------------------------------------------------
        //! \return A kernel executor.
        //-----------------------------------------------------------------------------
        template<
            typename TStream,
            typename TGridBlockExtents,
            typename TBlockThreadExtents>
        ALPAKA_FCT_HOST auto create(
            TGridBlockExtents const & gridBlockExtents,
            TBlockThreadExtents const & blockThreadExtents,
            TStream const & stream)
        -> ExecT<acc::AccT<TStream>>
        {
            return create(
                workdiv::BasicWorkDiv(
                    gridBlockExtents,
                    blockThreadExtents),
                stream);
        }
    }
}
