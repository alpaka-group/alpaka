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
                typename TExec,
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
            typename TExec>
        using ExecT = typename traits::exec::ExecType<TExec>::type;

        //-----------------------------------------------------------------------------
        //! \return An executor.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TStream,
            typename TWorkDiv>
        ALPAKA_FCT_HOST auto create(
            TWorkDiv const & workDiv,
            TStream & stream)
        -> ExecT<TAcc>
        {
            static_assert(
                dim::DimT<TWorkDiv>::value == dim::DimT<TAcc>::value,
                "The dimensions of the accelerator and the work division have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << BOOST_CURRENT_FUNCTION
                << " gridBlockExtents: " << workdiv::getWorkDiv<Grid, Blocks>(workDiv)
                << ", blockThreadExtents: " << workdiv::getWorkDiv<Block, Threads>(workDiv)
                << std::endl;
#endif
            // This checks for a valid work division that is also compliant with the maxima of the accelerator.
            if(!workdiv::isValidWorkDiv<TAcc>(dev::getDev(stream), workDiv))
            {
                throw std::runtime_error("The given work division is not valid or not supported by the " + acc::getAccName<TAcc>() + " accelerator!");
            }

            return
                ExecT<TAcc>(
                    workDiv,
                    stream);
        }
        //-----------------------------------------------------------------------------
        //! Creates an executor for the given accelerator with the work division given by grid block extents and block thread extents.
        //! The larger of the two dimensions specifies the executor dimension.
        //! \return An executor.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TStream,
            typename TGridBlockExtents,
            typename TBlockThreadExtents>
        ALPAKA_FCT_HOST auto create(
            TGridBlockExtents const & gridBlockExtents,
            TBlockThreadExtents const & blockThreadExtents,
            TStream & stream)
        -> ExecT<TAcc>
        {
            static_assert(
                (dim::DimT<TAcc>::value >= dim::DimT<TBlockThreadExtents>::value) && (dim::DimT<TAcc>::value >= dim::DimT<TGridBlockExtents>::value),
                "The dimension of the accelerator has to be larger or equal dimensionality than the grid block and block thread extents!");

            return
                create<TAcc>(
                    workdiv::BasicWorkDiv<
                        dim::Dim<dim::DimT<TAcc>::value>>(
                        gridBlockExtents,
                        blockThreadExtents),
                    stream);
        }
    }
}
