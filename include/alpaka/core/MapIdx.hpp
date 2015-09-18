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

#include <alpaka/vec/Vec.hpp>               // Vec
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC

#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //! Maps a linear index to a N dimensional index.
            //#############################################################################
            template<
                std::size_t TidxDimOut,
                std::size_t TidxDimIn>
            struct MapIdx;
            //#############################################################################
            //! Maps a linear index to a linear index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param Index Idx to be mapped.
                // \param Extent Spatial size to map the index to.
                // \return Vector of dimension TidxDimOut.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    Vec<dim::DimInt<1u>, TElem> const & idx,
                    Vec<dim::DimInt<1u>, TElem> const & extent)
                -> Vec<dim::DimInt<1u>, TElem>
                {
#if !defined(__CUDA_ARCH__)
                    boost::ignore_unused(extent);
#endif
                    return idx;
                }
            };
            //#############################################################################
            //! Maps a linear index to a 3 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                3u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param Index Idx to be mapped.
                // \param Extent Spatial size to map the index to.
                // \return Vector of dimension TidxDimOut.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    Vec<dim::DimInt<1u>, TElem> const & idx,
                    Vec<dim::DimInt<3u>, TElem> const & extent)
                -> Vec<dim::DimInt<3u>, TElem>
                {
                    auto const & idx1d(idx[0u]);
                    auto const xyExtentProd(extent[2u] * extent[1u]);
                    auto const & extentX(extent[2]);

                    return {
                        idx1d / xyExtentProd,
                        (idx1d % xyExtentProd) / extentX,
                        idx1d % extentX};
                }
            };
            //#############################################################################
            //! Maps a linear index to a 2 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                2u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param Index Idx to be mapped.
                // \param Extent Spatial size to map the index to.
                // \return Vector of dimension TidxDimOut.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    Vec<dim::DimInt<1u>, TElem> const & idx,
                    Vec<dim::DimInt<2u>, TElem> const & extent)
                -> Vec<dim::DimInt<2u>, TElem>
                {
                    auto const & idx1d(idx[0u]);
                    auto const & extentX(extent[1u]);

                    return {
                        idx1d / extentX,
                        idx1d % extentX};
                }
            };
            //#############################################################################
            //! Maps a 3 dimensional index to a linear index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                3u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param Index Idx to be mapped.
                // \param Extent Spatial size to map the index to.
                // \return Vector of dimension TidxDimOut.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    Vec<dim::DimInt<3u>, TElem> const & idx,
                    Vec<dim::DimInt<3u>, TElem> const & extent)
                -> Vec<dim::DimInt<1u>, TElem>
                {
                    return (idx[0u] * extent[1u] + idx[1u]) * extent[2u] + idx[2u];
                }
            };
            //#############################################################################
            //! Maps a 2 dimensional index to a linear index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                2u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param Index Idx to be mapped.
                // \param Extent Spatial size to map the index to.
                // \return Vector of dimension TidxDimOut.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    Vec<dim::DimInt<2u>, TElem> const & idx,
                    Vec<dim::DimInt<2u>, TElem> const & extent)
                -> Vec<dim::DimInt<1u>, TElem>
                {
                    return idx[0u] * extent[1u] + idx[1u];
                }
            };
        }

        //#############################################################################
        //! Maps a N dimensional index to a N dimensional position.
        //!
        //! \tparam TidxDimOut Dimension of the index vector to map to.
        //! \tparam TidxDimIn Dimension of the index vector to map from.
        //! \tparam TElem Type of the elements of the index vector to map from.
        //#############################################################################
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            std::size_t TidxDimOut,
            std::size_t TidxDimIn,
            typename TElem>
        ALPAKA_FN_HOST_ACC auto mapIdx(
            Vec<dim::DimInt<TidxDimIn>, TElem> const & idx,
            Vec<dim::DimInt<(TidxDimOut < TidxDimIn) ? TidxDimIn : TidxDimOut>, TElem> const & extent)
        -> Vec<dim::DimInt<TidxDimOut>, TElem>
        {
            return detail::MapIdx<
                TidxDimOut,
                TidxDimIn>
            ::mapIdx(
                idx,
                extent);
        }
    }
}
