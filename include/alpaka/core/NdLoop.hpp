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

#include <alpaka/traits/Dim.hpp>    // DimT

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC
#include <alpaka/core/Vec.hpp>      // Vec

#include <utility>                  // std::forward

namespace alpaka
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
            template<
                std::size_t TuiCurDim,
                typename TIndex,
                typename TExtentsVec,
                typename TFunctor,
                typename... TArgs>
            ALPAKA_FCT_HOST_ACC static auto ndLoop(
                TIndex & index,
                TExtentsVec const & extents,
                TFunctor && f,
                TArgs && ... args)
            -> void
            {
                static_assert(
                    dim::DimT<TIndex>::value > 0u,
                    "The dimension given to ndLoop has to be larger than zero!");
                static_assert(
                    dim::DimT<TIndex>::value == dim::DimT<TExtentsVec>::value,
                    "The dimensions of the iteration vector and the extents vector have to be identical!");
                static_assert(
                    dim::DimT<TIndex>::value > TuiCurDim,
                    "The current dimension has to be in the rang [0,dim-1]!");

                for(index[TuiCurDim] = 0u; index[TuiCurDim] < extents[TuiCurDim]; ++index[TuiCurDim])
                {
                    std::forward<TFunctor>(f)(index, std::forward<TArgs>(args)...);
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
            template<
                std::size_t TuiCurDim,
                typename TIndex,
                typename TExtentsVec,
                typename TFunctor,
                typename... TArgs>
            ALPAKA_FCT_HOST_ACC static auto ndLoop(
                TIndex & index,
                TExtentsVec const & extents,
                TFunctor && f,
                TArgs && ... args)
            -> void
            {
                static_assert(
                    dim::DimT<TIndex>::value > 0u,
                    "The dimension given to ndLoop has to be larger than zero!");
                static_assert(
                    dim::DimT<TIndex>::value == dim::DimT<TExtentsVec>::value,
                    "The dimensions of the iteration vector and the extents vector have to be identical!");
                static_assert(
                    dim::DimT<TIndex>::value > TuiCurDim,
                    "The current dimension has to be in the rang [0,dim-1]!");

                for(index[TuiCurDim] = 0u; index[TuiCurDim] < extents[TuiCurDim]; ++index[TuiCurDim])
                {
                    detail::NdLoop<
                        (TuiCurDim+2u == dim::DimT<TIndex>::value)>
                    ::template ndLoop<
                        TuiCurDim+1u>(
                            index,
                            extents,
                            std::forward<TFunctor>(f),
                            std::forward<TArgs>(args)...);
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
    template<
        typename TExtentsVec,
        typename TFunctor,
        typename... TArgs>
    ALPAKA_FCT_HOST_ACC auto ndLoop(
        TExtentsVec const & extents,
        TFunctor && f,
        TArgs && ... args)
    -> void
    {
        static_assert(
            dim::DimT<TExtentsVec>::value > 0u,
            "The dimension given to ndLoop has to be larger than zero!");

        auto vuiIdx(
            Vec<dim::DimT<TExtentsVec>>::zeros());

        detail::NdLoop<
            (dim::DimT<TExtentsVec>::value == 1u)>
        ::template ndLoop<
            0u>(
                vuiIdx,
                extents,
                std::forward<TFunctor>(f),
                std::forward<TArgs>(args)...);
    }
}
