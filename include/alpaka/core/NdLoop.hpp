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

#include <alpaka/dim/Traits.hpp>            // Dim

#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC
#include <alpaka/vec/Vec.hpp>               // Vec

#include <alpaka/core/IntegerSequence.hpp>  // core::detail::index_sequence

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
                typename TIndexSequence>
            struct NdLoop;
            //#############################################################################
            //! N-dimensional loop iteration template.
            //#############################################################################
            template<>
            struct NdLoop<
                core::detail::index_sequence<>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TIndex,
                    typename TExtentsVec,
                    typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentsVec const & extents,
                    TFnObj const & f)
                -> void
                {
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            //#############################################################################
            template<
                std::size_t Tdim>
            struct NdLoop<
                core::detail::index_sequence<Tdim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TIndex,
                    typename TExtentsVec,
                    typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentsVec const & extents,
                    TFnObj const & f)
                -> void
                {
                    static_assert(
                        dim::Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoopIncIdx has to be larger than zero!");
                    static_assert(
                        dim::Dim<TIndex>::value == dim::Dim<TExtentsVec>::value,
                        "The dimensions of the iteration vector and the extents vector have to be identical!");
                    static_assert(
                        dim::Dim<TIndex>::value > Tdim,
                        "The current dimension has to be in the rang [0,dim-1]!");

                    for(idx[Tdim] = 0u; idx[Tdim] < extents[Tdim]; ++idx[Tdim])
                    {
                        f(idx);
                    }
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            //#############################################################################
            template<
                std::size_t Tdim,
                std::size_t... Tdims>
            struct NdLoop<
                core::detail::index_sequence<Tdim, Tdims...>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TIndex,
                    typename TExtentsVec,
                    typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentsVec const & extents,
                    TFnObj const & f)
                -> void
                {
                    static_assert(
                        dim::Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoop has to be larger than zero!");
                    static_assert(
                        dim::Dim<TIndex>::value == dim::Dim<TExtentsVec>::value,
                        "The dimensions of the iteration vector and the extents vector have to be identical!");
                    static_assert(
                        dim::Dim<TIndex>::value > Tdim,
                        "The current dimension has to be in the rang [0,dim-1]!");

                    for(idx[Tdim] = 0u; idx[Tdim] < extents[Tdim]; ++idx[Tdim])
                    {
                        detail::NdLoop<
                            core::detail::index_sequence<Tdims...>>
                        ::template ndLoop(
                                idx,
                                extents,
                                f);
                    }
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested in the order given by the index_sequence with the first element being the outermost and the last index the innermost loop.
        //!
        //! \param indexSequence A sequence of indices being a permutation of the values [0, dim-1], where every values occurs at most once.
        //! \param extents N-dimensional loop extents.
        //! \param f The function called at each iteration.
        //! \param args,... The additional arguments given to each function call.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtentsVec,
            typename TFnObj,
            std::size_t... Tdims>
        ALPAKA_FN_HOST_ACC auto ndLoop(
            core::detail::index_sequence<Tdims...> const & indexSequence,
            TExtentsVec const & extents,
            TFnObj const & f)
        -> void
        {
            static_assert(
                dim::Dim<TExtentsVec>::value > 0u,
                "The dimension of the extents given to ndLoop has to be larger than zero!");
            static_assert(
                core::detail::IntegerSequenceValuesInRange<core::detail::index_sequence<Tdims...>, std::size_t, 0, dim::Dim<TExtentsVec>::value>::value,
                "The values in the index_sequence have to in the rang [0,dim-1]!");
            static_assert(
                core::detail::IntegerSequenceValuesUnique<core::detail::index_sequence<Tdims...>>::value,
                "The values in the index_sequence have to be unique!");

            auto idx(
                Vec<dim::Dim<TExtentsVec>, size::Size<TExtentsVec>>::zeros());

            detail::NdLoop<
                core::detail::index_sequence<Tdims...>>
            ::template ndLoop(
                    idx,
                    extents,
                    f);
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
            typename TFnObj>
        ALPAKA_FN_HOST_ACC auto ndLoopIncIdx(
            TExtentsVec const & extents,
            TFnObj const & f)
        -> void
        {
            ndLoop(
                core::detail::make_index_sequence<dim::Dim<TExtentsVec>::value>(),
                extents,
                f);
        }
    }
}
