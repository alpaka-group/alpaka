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

#include <alpaka/traits/Dev.hpp>            // dev::DevManT
#include <alpaka/traits/Acc.hpp>            // getAccDevProps

#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/core/ForEachType.hpp>      // forEachType
#include <alpaka/core/Vec.hpp>              // Vec
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST

#include <boost/mpl/vector.hpp>             // boost::mpl::vector

#include <cmath>                            // std::ceil
#include <algorithm>                        // std::min
#include <functional>                       // std::bind

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
            //! The maximum block thread extents correction wrapper.
            //#############################################################################
            struct CorrectMaxBlockThreadExtents
            {
                //-----------------------------------------------------------------------------
                //! Corrects the maximum block thread extents if it is larger then the one supported by the given accelerator type.
                //-----------------------------------------------------------------------------
                template<
                    typename TAcc>
                ALPAKA_FCT_HOST auto operator()(
                    Vec3<> & v3uiBlockThreadExtents)
                -> void
                {
                    auto const vDevs(dev::getDevs<dev::DevManT<TAcc>>());
                    for(auto const & dev : vDevs)
                    {
                        auto const devProps(acc::getAccDevProps<TAcc>(dev));
                        auto const & v3uiBlockThreadExtentsMax(devProps.m_v3uiBlockThreadExtentsMax);

                        v3uiBlockThreadExtents = Vec3<>(
                            std::min(v3uiBlockThreadExtents[0u], v3uiBlockThreadExtentsMax[0u]),
                            std::min(v3uiBlockThreadExtents[1u], v3uiBlockThreadExtentsMax[1u]),
                            std::min(v3uiBlockThreadExtents[2u], v3uiBlockThreadExtentsMax[2u]));
                    }
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block thread extents supported by all of the given accelerators.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq>
        ALPAKA_FCT_HOST auto getMaxBlockThreadExtentsAccsDevices()
        -> Vec3<>
        {
            static_assert(
                boost::mpl::is_sequence<TAccSeq>::value,
                "TAccSeq is required to be a mpl::sequence!");

            Vec3<> v3uiMaxBlockThreadExtents(
                std::numeric_limits<UInt>::max(),
                std::numeric_limits<UInt>::max(),
                std::numeric_limits<UInt>::max());

            forEachType<TAccSeq>(
                detail::CorrectMaxBlockThreadExtents(),
                std::ref(v3uiMaxBlockThreadExtents)
                );

            return v3uiMaxBlockThreadExtents;
        }

        namespace detail
        {
            //#############################################################################
            //! The maximum block thread count correction wrapper.
            //#############################################################################
            struct CorrectMaxBlockThreadCount
            {
                //-----------------------------------------------------------------------------
                //! Corrects the maximum block thread count if it is larger then the one supported by the given accelerator type.
                //-----------------------------------------------------------------------------
                template<
                    typename TAcc>
                ALPAKA_FCT_HOST auto operator()(
                    UInt & uiBlockThreadCount)
                -> void
                {
                    auto const vDevs(dev::getDevs<dev::DevManT<TAcc>>());
                    for(auto const & dev : vDevs)
                    {
                        auto const devProps(acc::getAccDevProps<TAcc>(dev));
                        auto const & uiBlockThreadCountMax(devProps.m_uiBlockThreadsCountMax);

                        uiBlockThreadCount = std::min(uiBlockThreadCount, uiBlockThreadCountMax);
                    }
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block thread count supported by all of the given accelerators.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq>
        ALPAKA_FCT_HOST auto getMaxBlockThreadCountAccsDevices()
        -> UInt
        {
            static_assert(boost::mpl::is_sequence<TAccSeq>::value, "TAccSeq is required to be a mpl::sequence!");

            UInt uiMaxBlockThreadCount(
                std::numeric_limits<UInt>::max());

            forEachType<TAccSeq>(
                detail::CorrectMaxBlockThreadCount(),
                std::ref(uiMaxBlockThreadCount)
                );

            return uiMaxBlockThreadCount;
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
            ALPAKA_FCT_HOST auto nextLowerOrEqualFactor(
                T const & uiMaxDivisor,
                T const & uiDividend)
            -> T
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
            //! Subdivides the given grid thread extents into blocks restricted by:
            //! 1. The maximum block thread extents
            //! 2. The maximum block thread count
            //! 3. The requirement of the block thread extents to divide the grid thread extents without remainder
            //!
            //! \param v3uiGridThreadExtents
            //!     The full extents of threads in the grid.
            //! \param v3uiMaxBlockThreadExtents
            //!     The maximum extents of threads in a block.
            //! \param uiMaxBlockThreadsCount
            //!     The maximum number of threads in a block.
            //! \param bRequireBlockThreadExtentsToDivideGridThreadExtents
            //!     If this is true, the grid thread extents will be multiples of the corresponding block thread extents.
            //!     NOTE: If v3uiGridThreadExtents is prime (or otherwise bad chosen) in a dimension, the block thread extent will be one in this dimension.
            //#############################################################################
            ALPAKA_FCT_HOST auto subdivideGridThreads(
                Vec3<> const & v3uiGridThreadExtents,
                Vec3<> const & v3uiMaxBlockThreadExtents,
                UInt const & uiMaxBlockThreadsCount,
                bool bRequireBlockThreadExtentsToDivideGridThreadExtents = true)
            -> BasicWorkDiv
            {
                assert(v3uiGridThreadExtents[0u]>0);
                assert(v3uiGridThreadExtents[1u]>0);
                assert(v3uiGridThreadExtents[2u]>0);

                // 1. Restrict the max block thread extents with the grid thread extents.
                // This removes dimensions not required in the given grid thread extents.
                // This has to be done before the uiMaxBlockThreadsCount clipping to get the maximum correctly.
                Vec3<> v3uiBlockThreadExtents(
                    std::min(v3uiMaxBlockThreadExtents[0u], v3uiGridThreadExtents[0u]),
                    std::min(v3uiMaxBlockThreadExtents[1u], v3uiGridThreadExtents[1u]),
                    std::min(v3uiMaxBlockThreadExtents[2u], v3uiGridThreadExtents[2u]));

                // 2. If the block thread extents require more threads then available on the accelerator, clip it.
                if(v3uiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
                {
                    // Begin in z dimension.
                    UInt uiDim(0u);
                    // Very primitive clipping. Just halve it until it fits repeatedly iterating over the dimensions.
                    while(v3uiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
                    {
                        v3uiBlockThreadExtents[uiDim] = std::max(static_cast<UInt>(1u), static_cast<UInt>(v3uiBlockThreadExtents[uiDim] / 2u));
                        uiDim = (uiDim+1u) % 3u;
                    }
                }

                if(bRequireBlockThreadExtentsToDivideGridThreadExtents)
                {
                    // Make the block thread extents divide the grid thread extents.
                    v3uiBlockThreadExtents = Vec3<>(
                        detail::nextLowerOrEqualFactor(v3uiBlockThreadExtents[0u], v3uiGridThreadExtents[0u]),
                        detail::nextLowerOrEqualFactor(v3uiBlockThreadExtents[1u], v3uiGridThreadExtents[1u]),
                        detail::nextLowerOrEqualFactor(v3uiBlockThreadExtents[2u], v3uiGridThreadExtents[2u]));
                }

                // Set the grid block extents (rounded to the next integer not less then the quotient.
                Vec3<> const v3uiGridBlockExtents(
                    static_cast<UInt>(std::ceil(static_cast<double>(v3uiGridThreadExtents[0u]) / static_cast<double>(v3uiBlockThreadExtents[0u]))),
                    static_cast<UInt>(std::ceil(static_cast<double>(v3uiGridThreadExtents[1u]) / static_cast<double>(v3uiBlockThreadExtents[1u]))),
                    static_cast<UInt>(std::ceil(static_cast<double>(v3uiGridThreadExtents[2u]) / static_cast<double>(v3uiBlockThreadExtents[2u]))));

                return BasicWorkDiv(v3uiGridBlockExtents, v3uiBlockThreadExtents);
            }
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAccs The accelerators for which this work division has to be valid.
        //! \param gridThreadExtents The full extents of threads in the grid.
        //! \param bRequireBlockThreadExtentsToDivideGridThreadExtents If the grid thread extents have to be a multiple of the block thread extents.
        //! \return The work division.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq,
            typename TExtents>
        ALPAKA_FCT_HOST auto getValidWorkDiv(
            TExtents const & gridThreadExtents = TExtents(),
            bool bRequireBlockThreadExtentsToDivideGridThreadExtents = true)
        -> BasicWorkDiv
        {
            static_assert(
                boost::mpl::is_sequence<TAccSeq>::value,
                "TAccSeq is required to be a mpl::sequence!");
            static_assert(
                dim::DimT<TExtents>::value <= 3,
                "TExtents is required to have less than or equal 3 dimensions!");

            return detail::subdivideGridThreads(
                extent::getExtentsNd<dim::Dim3, UInt>(gridThreadExtents),
                getMaxBlockThreadExtentsAccsDevices<TAccSeq>(),
                getMaxBlockThreadCountAccsDevices<TAccSeq>(),
                bRequireBlockThreadExtentsToDivideGridThreadExtents);
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAcc The accelerator to test the validity on.
        //! \param dev The device to test the work div to for validity on.
        //! \param workDiv The work div to test for validity.
        //! \return If the work division is valid on this accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TDev,
            typename TWorkDiv>
        ALPAKA_FCT_HOST auto isValidWorkDiv(
            TDev const & dev,
            TWorkDiv const & workDiv)
        -> bool
        {
            auto const v3uiGridBlockExtents(getWorkDiv<Grid, Blocks, dim::Dim3>(workDiv));
            auto const v3uiBlockThreadExtents(getWorkDiv<Block, Threads, dim::Dim3>(workDiv));

            auto const devProps(acc::getAccDevProps<TAcc>(dev));
            auto const & v3uiBlockThreadExtentsMax(devProps.m_v3uiBlockThreadExtentsMax);
            auto const & uiBlockThreadCountMax(devProps.m_uiBlockThreadsCountMax);

            return !((v3uiGridBlockExtents[0] == 0)
                || (v3uiGridBlockExtents[1] == 0)
                || (v3uiGridBlockExtents[2] == 0)
                || (v3uiBlockThreadExtents[0] == 0)
                || (v3uiBlockThreadExtents[1] == 0)
                || (v3uiBlockThreadExtents[2] == 0)
                || (v3uiBlockThreadExtentsMax[0] < v3uiBlockThreadExtents[0])
                || (v3uiBlockThreadExtentsMax[1] < v3uiBlockThreadExtents[1])
                || (v3uiBlockThreadExtentsMax[2] < v3uiBlockThreadExtents[2])
                || (uiBlockThreadCountMax < v3uiBlockThreadExtents.prod()));
        }
    }
}
