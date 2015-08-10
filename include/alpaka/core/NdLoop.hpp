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

#include <alpaka/dim/Traits.hpp>    // Dim

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST_ACC
#include <alpaka/core/Vec.hpp>      // Vec

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //! N-dimensional loop iteration template.
            //#############################################################################
            template<
                bool TbLastLoop>
            struct NdLoop;
            //#############################################################################
            //! N-dimensional loop iteration template.
            //#############################################################################
            template<>
            struct NdLoop<
                true>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    std::size_t TuiCurDim,
                    typename TIndex,
                    typename TExtentsVec,
                    typename TFnObj,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentsVec const & extents,
                    TFnObj const & f,
                    TArgs const & ... args)
                -> void
                {
                    static_assert(
                        dim::Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoop has to be larger than zero!");
                    static_assert(
                        dim::Dim<TIndex>::value == dim::Dim<TExtentsVec>::value,
                        "The dimensions of the iteration vector and the extents vector have to be identical!");
                    static_assert(
                        dim::Dim<TIndex>::value > TuiCurDim,
                        "The current dimension has to be in the rang [0,dim-1]!");

                    for(idx[TuiCurDim] = 0u; idx[TuiCurDim] < extents[TuiCurDim]; ++idx[TuiCurDim])
                    {
                        f(idx, args...);
                    }
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            //#############################################################################
            template<>
            struct NdLoop<
                false>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    std::size_t TuiCurDim,
                    typename TIndex,
                    typename TExtentsVec,
                    typename TFnObj,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentsVec const & extents,
                    TFnObj const & f,
                    TArgs const & ... args)
                -> void
                {
                    static_assert(
                        dim::Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoop has to be larger than zero!");
                    static_assert(
                        dim::Dim<TIndex>::value == dim::Dim<TExtentsVec>::value,
                        "The dimensions of the iteration vector and the extents vector have to be identical!");
                    static_assert(
                        dim::Dim<TIndex>::value > TuiCurDim,
                        "The current dimension has to be in the rang [0,dim-1]!");

                    for(idx[TuiCurDim] = 0u; idx[TuiCurDim] < extents[TuiCurDim]; ++idx[TuiCurDim])
                    {
                        detail::NdLoop<
                            (TuiCurDim+2u == dim::Dim<TIndex>::value)>
                        ::template ndLoop<
                            TuiCurDim+1u>(
                                idx,
                                extents,
                                f,
                                args...);
                    }
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested from index zero outmost to index (dim-1) innermost.
        //!
        //! \param extents N-dimensional loop extents.
        //! \param f The function called at each iteration.
        //! \param args,... The additional arguments given to each function call.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtentsVec,
            typename TFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto ndLoop(
            TExtentsVec const & extents,
            TFnObj const & f,
            TArgs const & ... args)
        -> void
        {
            static_assert(
                dim::Dim<TExtentsVec>::value > 0u,
                "The dimension given to ndLoop has to be larger than zero!");

            auto idx(
                Vec<dim::Dim<TExtentsVec>, size::Size<TExtentsVec>>::zeros());

            detail::NdLoop<
                (dim::Dim<TExtentsVec>::value == 1u)>
            ::template ndLoop<
                0u>(
                    idx,
                    extents,
                    f,
                    args...);
        }
    }
}
