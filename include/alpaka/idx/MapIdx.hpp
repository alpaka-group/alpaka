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

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

namespace alpaka
{
    namespace idx
    {
        namespace detail
        {
            //#############################################################################
            //! Maps a linear index to a N dimensional index.
            //#############################################################################
            template<
                std::size_t TidxDimOut,
                std::size_t TidxDimIn,
                typename TSfinae = void>
            struct MapIdx;
            //#############################################################################
            //! Maps a 1 dimensional index to a 1 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 1 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<1u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<1u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<1u>, TElem>
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    boost::ignore_unused(extent);
#endif
                    return idx;
                }
            };
            //#############################################################################
            //! Maps a 1 dimensional index to a 2 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                2u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 2 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<1u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<2u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<2u>, TElem>
                {
                    TElem const & idx1d(idx[0u]);
                    TElem const & extentX(extent[1u]);

                    return {
                        static_cast<TElem>(idx1d / extentX),
                        static_cast<TElem>(idx1d % extentX)};
                }
            };
            //#############################################################################
            //! Maps a 1 dimensional index to a 3 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                3u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 3 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<1u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<3u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<3u>, TElem>
                {
                    TElem const & idx1d(idx[0u]);
                    TElem const & extentX(extent[2]);
                    TElem const xyExtentProd(extent[2u] * extent[1u]);

                    return {
                        static_cast<TElem>(idx1d / xyExtentProd),
                        static_cast<TElem>((idx1d % xyExtentProd) / extentX),
                        static_cast<TElem>(idx1d % extentX)};
                }
            };
            //#############################################################################
            //! Maps a 1 dimensional index to a 4 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                4u,
                1u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 4 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<1u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<4u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<4u>, TElem>
                {
                    TElem const & idx1d(idx[0u]);
                    TElem const & extentX(extent[3]);
                    TElem const xyExtentProd(extent[3u] * extent[2u]);
                    TElem const xyzExtentProd(xyExtentProd * extent[1u]);

                    return {
                        static_cast<TElem>(idx1d / xyzExtentProd),
                        static_cast<TElem>((idx1d % xyzExtentProd) / xyExtentProd),
                        static_cast<TElem>((idx1d % xyExtentProd) / extentX),
                        static_cast<TElem>(idx1d % extentX)};
                }
            };
            //#############################################################################
            //! Maps a 2 dimensional index to a 1 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                2u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 1 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<2u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<2u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<1u>, TElem>
                {
                    return {
                        idx[0u] * extent[1u] + idx[1u]};
                }
            };
            //#############################################################################
            //! Maps a 3 dimensional index to a 1 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                3u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 1 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<3u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<3u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<1u>, TElem>
                {
                    return {
                        (idx[0u] * extent[1u] + idx[1u]) * extent[2u] + idx[2u]};
                }
            };
            //#############################################################################
            //! Maps a 4 dimensional index to a 1 dimensional index.
            //#############################################################################
            template<>
            struct MapIdx<
                1u,
                4u>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 1 dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<4u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<4u>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<1u>, TElem>
                {
                    return {
                        ((idx[0u] * extent[1u] + idx[1u]) * extent[2u] + idx[2u]) * extent[3u] + idx[3u]};
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
            vec::Vec<dim::DimInt<TidxDimIn>, TElem> const & idx,
            vec::Vec<dim::DimInt<(TidxDimOut < TidxDimIn) ? TidxDimIn : TidxDimOut>, TElem> const & extent)
        -> vec::Vec<dim::DimInt<TidxDimOut>, TElem>
        {
            return
                detail::MapIdx<
                    TidxDimOut,
                    TidxDimIn>
                ::mapIdx(
                    idx,
                    extent);
        }
    }
}
