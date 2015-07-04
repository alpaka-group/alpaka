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

#include <alpaka/acc/Traits.hpp>            // acc::getAccName
#include <alpaka/dev/Traits.hpp>            // dev::getDev
#include <alpaka/workdiv/Traits.hpp>        // workdiv::getWorkDiv
#include <alpaka/size/Traits.hpp>           // size::SizeT

#include <alpaka/core/WorkDivHelpers.hpp>   // workdiv::isValidWorkDiv
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST

#include <type_traits>                      // std::is_same

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The executor specifics.
    //-----------------------------------------------------------------------------
    namespace exec
    {
        //-----------------------------------------------------------------------------
        //! The execution traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The executor type trait.
            //#############################################################################
            template<
                typename TExec,
                typename TSfinae = void>
            struct ExecType;
        }

        //#############################################################################
        //! The executor type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TExec>
        using ExecT = typename traits::ExecType<TExec>::type;

        //-----------------------------------------------------------------------------
        //! \return An executor.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TStream,
            typename TWorkDiv>
        ALPAKA_FN_HOST auto create(
            TWorkDiv const & workDiv,
            TStream & stream)
        -> ExecT<TAcc>
        {
            static_assert(
                dim::DimT<TWorkDiv>::value == dim::DimT<TAcc>::value,
                "The dimensions of TAcc and TWorkDiv have to be identical!");
            static_assert(
                std::is_same<size::SizeT<TWorkDiv>, size::SizeT<TAcc>>::value,
                "The size type of TAcc and the size type of TWorkDiv have to be identical!");

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
        ALPAKA_FN_HOST auto create(
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
                    workdiv::WorkDivMembers<
                        dim::Dim<dim::DimT<TAcc>::value>,
                        size::SizeT<TAcc>>(
                            gridBlockExtents,
                            blockThreadExtents),
                    stream);
        }
    }
}
