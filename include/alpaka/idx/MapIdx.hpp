/**
* \file
* Copyright 2014-2017 Benjamin Worpitz, Axel Huebl
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
                bool isSame = ( TidxDimOut == TidxDimIn ),
                typename TSfinae = void>
            struct MapIdx;
            //#############################################################################
            //! Maps a N dimensional index to the same N dimensional index.
            //#############################################################################
            template<
                std::size_t TidxDim>
            struct MapIdx<
                TidxDim,
                TidxDim,
                true>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A N dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<TidxDim>, TElem> const & idx,
#if !BOOST_ARCH_CUDA_DEVICE
                    vec::Vec<dim::DimInt<TidxDim>, TElem> const & extent)
#else
                    vec::Vec<dim::DimInt<TidxDim>, TElem> const &)
#endif
                -> vec::Vec<dim::DimInt<TidxDim>, TElem>
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    boost::ignore_unused(extent);
#endif
                    return idx;
                }
            };
            //#############################################################################
            //! Maps a 1 dimensional index to a N dimensional index.
            //#############################################################################
            template<
                std::size_t TidxDimOut>
            struct MapIdx<
                TidxDimOut,
                1u,
                false>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to
                // \return A N dimensional vector.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<1u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<TidxDimOut>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<TidxDimOut>, TElem>
                {
                    // wrong constructor!
                    using MyVec = vec::Vec<dim::DimInt<TidxDimOut>, TElem>;
                    MyVec idxnd( MyVec::all( 0u ) );
                    MyVec hyperPlanesBefore( MyVec::all( 1u ) );

                    for( std::size_t d = 1u; d < TidxDimIn; ++d )
                        hyperPlanesBefore[d] = hyperPlanesBefore[d-1] * extent[d-1];

                    for( std::size_t d = 0u; d < TidxDimIn; ++d )
                    {
                        // optimization 1: % can be skipped for (d == TidxDimIn - 1u)
                        // optimization 2: hyperPlanesBefore[d] is 1u for (d == 0u)
                        idxnd[d] = idx[d] / hyperPlanesBefore[d] % extent[d];
                    }
                    return idxnd;
                }
            };
            //#############################################################################
            //! Maps a N dimensional index to a 1 dimensional index.
            //#############################################################################
            template<
                std::size_t TidxDimIn>
            struct MapIdx<
                1u,
                TidxDimIn,
                false>
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
                    vec::Vec<dim::DimInt<TidxDimIn>, TElem> const & idx,
                    vec::Vec<dim::DimInt<TidxDimIn>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<1u>, TElem>
                {
                    std::size_t idx1d( 0u );
                    std::size_t hyperPlanesVolume( 1u );
                    for( std::size_t d = 0u; d < TidxDimIn; ++d )
                    {
                        idx1d += idx[d] * hyperPlanesVolume;
                        hyperPlanesVolume *= extent[d];
                    }
                    return {
                        static_cast<TElem>( idx1d )};
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
