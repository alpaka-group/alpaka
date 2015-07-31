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
#include <alpaka/size/Traits.hpp>           // size::Size

#include <alpaka/core/WorkDivHelpers.hpp>   // workdiv::isValidWorkDiv
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST

#include <type_traits>                      // std::is_same, std::decay

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
                typename TKernelFnObj,
                typename... TArgs/*,
                typename TSfinae = void*/>
            struct ExecType;
        }

        //#############################################################################
        //! The executor type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TExec,
            typename TKernelFnObj,
            typename... TArgs>
        using Exec = typename traits::ExecType<TExec, TKernelFnObj, TArgs...>::type;

        //-----------------------------------------------------------------------------
        //! \return An executor.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST auto create(
            TWorkDiv && workDiv,
            TKernelFnObj && kernelFnObj,
            TArgs && ... args)
        -> Exec<
            TAcc,
            typename std::decay<TKernelFnObj>::type,
            typename std::decay<TArgs>::type...>
        {
            static_assert(
                dim::Dim<typename std::decay<TWorkDiv>::type>::value == dim::Dim<TAcc>::value,
                "The dimensions of TAcc and TWorkDiv have to be identical!");
            static_assert(
                std::is_same<size::Size<typename std::decay<TWorkDiv>::type>, size::Size<TAcc>>::value,
                "The size type of TAcc and the size type of TWorkDiv have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << BOOST_CURRENT_FUNCTION
                << " gridBlockExtents: " << workdiv::getWorkDiv<Grid, Blocks>(workDiv)
                << ", blockThreadExtents: " << workdiv::getWorkDiv<Block, Threads>(workDiv)
                << std::endl;
#endif
/*#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            // This checks for a valid work division that is also compliant with the maxima of the accelerator.
            if(!workdiv::isValidWorkDiv<TAcc>(dev::getDev(stream), workDiv))
            {
                throw std::runtime_error("The given work division is not valid or not supported by the " + acc::getAccName<TAcc>() + " accelerator!");
            }
#endif*/

            return
                Exec<
                    TAcc,
                    typename std::decay<TKernelFnObj>::type,
                    typename std::decay<TArgs>::type...>(
                        std::forward<TWorkDiv>(workDiv),
                        std::forward<TKernelFnObj>(kernelFnObj),
                        std::forward<TArgs>(args)...);
        }
        //-----------------------------------------------------------------------------
        //! Creates an executor for the given accelerator with the work division given by grid block extents and block thread extents.
        //! The larger of the two dimensions specifies the executor dimension.
        //! \return An executor.
        //-----------------------------------------------------------------------------
        /*template<
            typename TAcc,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename TGridBlockExtents,
            typename TBlockThreadExtents>
        ALPAKA_FN_HOST auto create(
            TGridBlockExtents const & gridBlockExtents,
            TBlockThreadExtents const & blockThreadExtents,
            TKernelFnObj && kernelFnObj,
            TArgs && ... args)
        -> Exec<
            typename std::decay<TAcc>::type,
            typename std::decay<TKernelFnObj>::type,
            typename std::decay<TArgs>::type...>
        {
            static_assert(
                (dim::Dim<TAcc>::value >= dim::Dim<TBlockThreadExtents>::value) && (dim::Dim<TAcc>::value >= dim::Dim<TGridBlockExtents>::value),
                "The dimension of the accelerator has to be larger or equal dimensionality than the grid block and block thread extents!");

            return
                create<TAcc>(
                    workdiv::WorkDivMembers<
                        dim::DimInt<dim::Dim<TAcc>::value>,
                        size::Size<TAcc>>(
                            gridBlockExtents,
                            blockThreadExtents),
                    std::forward<TKernelFnObj>(kernelFnObj),
                    std::forward<TArgs>(args)...);
        }*/
    }
}
