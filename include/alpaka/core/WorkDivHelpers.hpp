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

#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>              // Vec
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv

#include <alpaka/traits/Device.hpp>         // dev::DevManT, getDevProps

#include <cmath>                            // std::ceil
#include <algorithm>                        // std::min
#include <functional>                       // std::bind

#include <boost/mpl/for_each.hpp>           // boost::mpl::for_each
#include <boost/mpl/vector.hpp>             // boost::mpl::vector

//-----------------------------------------------------------------------------
//! The alpaka library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    namespace workdiv
    {
        namespace detail
        {
            //#############################################################################
            //! The maximum block kernels extents correction wrapper.
            //#############################################################################
            struct CorrectMaxBlockKernelExtents
            {
                //-----------------------------------------------------------------------------
                //! Corrects the maximum block kernels extents if it is larger then the one supported by the given accelerator type.
                //-----------------------------------------------------------------------------
                template<
                    typename TAcc>
                ALPAKA_FCT_HOST void operator()(
                    TAcc const &,
                    Vec<3u> & v3uiBlockKernelExtents)
                {
                    auto const devProps(dev::getDevProps(dev::DevManT<TAcc>::getCurrentDevice()));
                    auto const & v3uiBlockKernelsExtentsMax(devProps.m_v3uiBlockKernelsExtentsMax);

                    v3uiBlockKernelExtents = Vec<3u>(
                        std::min(v3uiBlockKernelExtents[0u], v3uiBlockKernelsExtentsMax[0u]),
                        std::min(v3uiBlockKernelExtents[1u], v3uiBlockKernelsExtentsMax[1u]),
                        std::min(v3uiBlockKernelExtents[2u], v3uiBlockKernelsExtentsMax[2u]));
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block kernels extents supported by all of the given accelerators.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq>
        ALPAKA_FCT_HOST Vec<3u> getMaxBlockKernelExtentsAccelerators()
        {
            static_assert(boost::mpl::is_sequence<TAccSeq>::value, "TAccSeq is required to be a mpl::sequence!");

            Vec<3u> v3uiMaxBlockKernelExtents(
                std::numeric_limits<Vec<3u>::Value>::max(),
                std::numeric_limits<Vec<3u>::Value>::max(),
                std::numeric_limits<Vec<3u>::Value>::max());

            boost::mpl::for_each<TAccSeq>(
                std::bind(detail::CorrectMaxBlockKernelExtents(), std::placeholders::_1, std::ref(v3uiMaxBlockKernelExtents))
                );

            return v3uiMaxBlockKernelExtents;
        }

        namespace detail
        {
            //#############################################################################
            //! The maximum block kernels count correction wrapper.
            //#############################################################################
            struct CorrectMaxBlockKernelCount
            {
                //-----------------------------------------------------------------------------
                //! Corrects the maximum block kernels count if it is larger then the one supported by the given accelerator type.
                //-----------------------------------------------------------------------------
                template<
                    typename TAcc>
                ALPAKA_FCT_HOST void operator()(
                    TAcc const &,
                    std::size_t & uiBlockKernelCount)
                {
                    auto const devProps(dev::getDevProps(dev::DevManT<TAcc>::getCurrentDevice()));
                    auto const & uiBlockKernelCountMax(devProps.m_uiBlockKernelsCountMax);

                    uiBlockKernelCount = std::min(uiBlockKernelCount, uiBlockKernelCountMax);
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block kernels count supported by all of the given accelerators.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq>
        ALPAKA_FCT_HOST std::size_t getMaxBlockKernelCountAccelerators()
        {
            static_assert(boost::mpl::is_sequence<TAccSeq>::value, "TAccSeq is required to be a mpl::sequence!");

            std::size_t uiMaxBlockKernelCount(
                std::numeric_limits<std::size_t>::max());

            boost::mpl::for_each<TAccSeq>(
                std::bind(detail::CorrectMaxBlockKernelCount(), std::placeholders::_1, std::ref(uiMaxBlockKernelCount))
                );

            return uiMaxBlockKernelCount;
        }

        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! \param uiMaxDivisor The maximum divisor.
            //! \param uiDividend The dividend.
            //! \return The biggest number that satisfies the following conditions:
            //!     1) uiDividend/ret==0
            //!     2) ret<=uiMaxDivisor
            //-----------------------------------------------------------------------------
            template<
                typename T,
                typename = typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            ALPAKA_FCT_HOST T nextLowerOrEqualFactor(
                T const & uiMaxDivisor, 
                T const & uiDividend)
            {
                T uiDivisor(uiMaxDivisor);
                // \TODO: This is not very efficient. Replace with a better algorithm.
                while((uiDividend%uiDivisor) != 0)
                {
                    --uiDivisor;
                }
                return uiDivisor;
            }

            //#############################################################################
            //! Subdivides the given grid kernels extents into blocks restricted by:
            //! 1. the maximum block kernels extents and 
            //! 2. the maximum block kernels count.
            //!
            //! \param v3uiGridKernelsExtents
            //!     The full extents of kernels in the grid.
            //! \param v3uiMaxBlockKernelsExtents
            //!     The maximum extents of kernels in a block.
            //! \param uiMaxBlockKernelsCount
            //!     The maximum number of kernels in a block.
            //! \param bRequireBlockKernelsExtentsToDivideGridKernelsExtents
            //!     If this is true, the grid kernels extents will be multiples of the corresponding block kernels extents.
            //!     NOTE: If v3uiGridKernelsExtents is prime (or otherwise bad chosen) in a dimension, the block kernels extent will be one in this dimension.
            //! \param bUniformBlockKernelsExtentsClipping
            //!     If this is true, the block kernels extents will be clipped uniformly.
            //!     This means that all values of the extent will be processed uniformly at the same time.
            //!     This can lead to smaller blocks but allows to keep the ratio between dimensions (in some limits due to integer rounding).
            //#############################################################################
            ALPAKA_FCT_HOST BasicWorkDiv subdivideGridKernels(
                Vec<3u> const & v3uiGridKernelsExtents,
                Vec<3u> const & v3uiMaxBlockKernelsExtents,
                std::size_t const & uiMaxBlockKernelsCount,
                bool bRequireBlockKernelsExtentsToDivideGridKernelsExtents = true)
            {
                assert(v3uiGridKernelsExtents[0u]>0);
                assert(v3uiGridKernelsExtents[1u]>0);
                assert(v3uiGridKernelsExtents[2u]>0);

                // 1. Restrict the max block kernels extents with the grid kernels extents.
                // This removes dimensions not required in the given gird kernels extents.
                // This has to be done before the uiMaxBlockKernelsCount clipping to get the maximum correctly.
                Vec<3u> v3uiBlockKernelsExtents(
                    std::min(v3uiMaxBlockKernelsExtents[0u], v3uiGridKernelsExtents[0u]),
                    std::min(v3uiMaxBlockKernelsExtents[1u], v3uiGridKernelsExtents[1u]),
                    std::min(v3uiMaxBlockKernelsExtents[2u], v3uiGridKernelsExtents[2u]));

                // 2. If the block kernels extents require more kernels then available on the accelerator, clip it.
                if(v3uiBlockKernelsExtents.prod() > uiMaxBlockKernelsCount)
                {
                    // Begin in z dimension.
                    std::size_t uiDim(2);
                    // Very primitive clipping. Just halve it until it fits repeatedly iterating over the dimensions.
                    while(v3uiBlockKernelsExtents.prod() > uiMaxBlockKernelsCount)
                    {
                        v3uiBlockKernelsExtents[uiDim] = std::max(static_cast<Vec<3u>::Value>(1u), static_cast<Vec<3u>::Value>(v3uiBlockKernelsExtents[uiDim] / 2u));
                        uiDim = (uiDim+2) % 3;
                    }
                }

                if(bRequireBlockKernelsExtentsToDivideGridKernelsExtents)
                {
                    // Make the block kernels extents divide the grid kernels extents.
                    v3uiBlockKernelsExtents = Vec<3u>(
                        detail::nextLowerOrEqualFactor(v3uiBlockKernelsExtents[0u], v3uiGridKernelsExtents[0u]),
                        detail::nextLowerOrEqualFactor(v3uiBlockKernelsExtents[1u], v3uiGridKernelsExtents[1u]),
                        detail::nextLowerOrEqualFactor(v3uiBlockKernelsExtents[2u], v3uiGridKernelsExtents[2u]));
                }

                // Set the grid blocks extents (rounded to the next integer not less then the quotient.
                Vec<3u> const v3uiGridBlocksExtents(
                    static_cast<Vec<3u>::Value>(std::ceil(static_cast<double>(v3uiGridKernelsExtents[0u]) / static_cast<double>(v3uiBlockKernelsExtents[0u]))),
                    static_cast<Vec<3u>::Value>(std::ceil(static_cast<double>(v3uiGridKernelsExtents[1u]) / static_cast<double>(v3uiBlockKernelsExtents[1u]))),
                    static_cast<Vec<3u>::Value>(std::ceil(static_cast<double>(v3uiGridKernelsExtents[2u]) / static_cast<double>(v3uiBlockKernelsExtents[2u]))));

                return BasicWorkDiv(v3uiGridBlocksExtents, v3uiBlockKernelsExtents);
            }
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAccs The accelerators for which this work division has to be valid.
        //! \param v3uiGridKernelsExtents The full extents of kernels in the grid.
        //! \return The work division.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq>
        ALPAKA_FCT_HOST BasicWorkDiv getValidWorkDiv(
            Vec<3u> const & v3uiGridKernelsExtents,
            bool bRequireBlockKernelsExtentsToDivideGridKernelsExtents = true)
        {
            static_assert(boost::mpl::is_sequence<TAccSeq>::value, "TAccSeq is required to be a mpl::sequence!");

            return detail::subdivideGridKernels(
                v3uiGridKernelsExtents,
                getMaxBlockKernelExtentsAccelerators<TAccSeq>(),
                getMaxBlockKernelCountAccelerators<TAccSeq>(),
                bRequireBlockKernelsExtentsToDivideGridKernelsExtents);
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAcc The accelerator to test the validity on.
        //! \param workDiv The work div to test for validity.
        //! \return If the work division is valid on this accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TWorkDiv>
        ALPAKA_FCT_HOST bool isValidWorkDiv(
            TWorkDiv const & workDiv)
        {
            auto const v3uiGridBlocksExtents(getWorkDiv<Grid, Blocks, dim::Dim3>(workDiv));
            auto const v3uiBlockKernelsExtents(getWorkDiv<Block, Kernels, dim::Dim3>(workDiv));

            auto const devProps(dev::getDevProps(dev::DevManT<TAcc>::getCurrentDevice()));
            auto const & v3uiBlockKernelsExtentsMax(devProps.m_v3uiBlockKernelsExtentsMax);
            auto const & uiBlockKernelCountMax(devProps.m_uiBlockKernelsCountMax);

            return !((v3uiGridBlocksExtents[0] == 0)
                || (v3uiGridBlocksExtents[1] == 0)
                || (v3uiGridBlocksExtents[2] == 0)
                || (v3uiBlockKernelsExtents[0] == 0)
                || (v3uiBlockKernelsExtents[1] == 0)
                || (v3uiBlockKernelsExtents[2] == 0)
                || (v3uiBlockKernelsExtentsMax[0] < v3uiBlockKernelsExtents[0])
                || (v3uiBlockKernelsExtentsMax[1] < v3uiBlockKernelsExtents[1])
                || (v3uiBlockKernelsExtentsMax[2] < v3uiBlockKernelsExtents[2])
                || (uiBlockKernelCountMax < v3uiBlockKernelsExtents.prod()));
        }
    }
}
