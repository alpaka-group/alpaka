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
                    typename TAcc,
                    typename TDim>
                ALPAKA_FCT_HOST auto operator()(
                    Vec<TDim> & vuiBlockThreadExtents)
                -> void
                {
                    static_assert(
                        TDim::value == dim::DimT<TAcc>::value,
                        "The accelerator is required to have the same dimension as the block thread extents");

                    auto const vDevs(dev::getDevs<dev::DevManT<TAcc>>());
                    for(auto const & dev : vDevs)
                    {
                        auto const devProps(acc::getAccDevProps<TAcc>(dev));
                        auto const & vuiBlockThreadExtentsMax(devProps.m_vuiBlockThreadExtentsMax);

                        for(std::size_t i(0); i<TDim::value; ++i)
                        {
                            vuiBlockThreadExtents[i] = std::min(vuiBlockThreadExtents[i], vuiBlockThreadExtentsMax[i]);
                        }
                    }
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block thread extents supported by all of the given accelerators.
        //-----------------------------------------------------------------------------
        template<
            typename TAccSeq,
            typename TAccDim>
        ALPAKA_FCT_HOST auto getMaxBlockThreadExtentsAccsDevices()
        -> Vec<TAccDim>
        {
            static_assert(
                boost::mpl::is_sequence<TAccSeq>::value,
                "TAccSeq is required to be a mpl::sequence!");

            auto vuiMaxBlockThreadExtents(
                Vec<TAccDim>::all(
                    std::numeric_limits<UInt>::max()));

            forEachType<TAccSeq>(
                detail::CorrectMaxBlockThreadExtents(),
                vuiMaxBlockThreadExtents);

            return vuiMaxBlockThreadExtents;
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
                uiMaxBlockThreadCount);

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
            //! \param vuiGridThreadExtents
            //!     The full extents of threads in the grid.
            //! \param vuiMaxBlockThreadExtents
            //!     The maximum extents of threads in a block.
            //! \param uiMaxBlockThreadsCount
            //!     The maximum number of threads in a block.
            //! \param bRequireBlockThreadExtentsToDivideGridThreadExtents
            //!     If this is true, the grid thread extents will be multiples of the corresponding block thread extents.
            //!     NOTE: If vuiGridThreadExtents is prime (or otherwise bad chosen) in a dimension, the block thread extent will be one in this dimension.
            //#############################################################################
            template<
                typename TDim>
            ALPAKA_FCT_HOST auto subdivideGridThreads(
                Vec<TDim> const & vuiGridThreadExtents,
                Vec<TDim> const & vuiMaxBlockThreadExtents,
                UInt const & uiMaxBlockThreadsCount,
                bool bRequireBlockThreadExtentsToDivideGridThreadExtents = true)
            -> BasicWorkDiv<TDim>
            {
                assert(uiMaxBlockThreadsCount>0);
                for(std::size_t i(0); i<TDim::value; ++i)
                {
                    assert(vuiGridThreadExtents[i]>0);
                }

                // 1. Restrict the max block thread extents with the grid thread extents.
                // This removes dimensions not required in the given grid thread extents.
                // This has to be done before the uiMaxBlockThreadsCount clipping to get the maximum correctly.
                auto vuiBlockThreadExtents(Vec<TDim>::ones());
                for(std::size_t i(0); i<TDim::value; ++i)
                {
                    vuiBlockThreadExtents[i] = std::min(vuiMaxBlockThreadExtents[i], vuiGridThreadExtents[i]);
                }

                // 2. If the block thread extents require more threads then available on the accelerator, clip it.
                if(vuiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
                {
                    // Begin in last dimension.
                    UInt uiDim(0u);
                    // Very primitive clipping. Just halve it until it fits repeatedly iterating over the dimensions.
                    while(vuiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
                    {
                        vuiBlockThreadExtents[uiDim] = std::max(static_cast<UInt>(1u), static_cast<UInt>(vuiBlockThreadExtents[uiDim] / 2u));
                        uiDim = (uiDim+1u) % TDim::value;
                    }
                }

                if(bRequireBlockThreadExtentsToDivideGridThreadExtents)
                {
                    // Make the block thread extents divide the grid thread extents.
                    for(std::size_t i(0); i<TDim::value; ++i)
                    {
                        vuiBlockThreadExtents[i] = detail::nextLowerOrEqualFactor(vuiBlockThreadExtents[i], vuiGridThreadExtents[i]);
                    }
                }

                // Set the grid block extents (rounded to the next integer not less then the quotient.
                auto vuiGridBlockExtents(Vec<TDim>::ones());
                for(std::size_t i(0); i<TDim::value; ++i)
                {
                    vuiGridBlockExtents[i] = static_cast<UInt>(std::ceil(static_cast<double>(vuiGridThreadExtents[0u]) / static_cast<double>(vuiBlockThreadExtents[0u])));
                }

                return BasicWorkDiv<TDim>(vuiGridBlockExtents, vuiBlockThreadExtents);
            }
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAccSeq The accelerator sequence for which this work division has to be valid.
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
        -> BasicWorkDiv<dim::DimT<TExtents>>
        {
            static_assert(
                boost::mpl::is_sequence<TAccSeq>::value,
                "TAccSeq is required to be a mpl::sequence!");
            return detail::subdivideGridThreads(
                extent::getExtentsVec<UInt>(gridThreadExtents),
                getMaxBlockThreadExtentsAccsDevices<TAccSeq, dim::DimT<TExtents>>(),
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
            auto const vuiGridBlockExtents(getWorkDiv<Grid, Blocks>(workDiv));
            auto const vuiBlockThreadExtents(getWorkDiv<Block, Threads>(workDiv));

            auto const devProps(acc::getAccDevProps<TAcc>(dev));
            auto const vuiBlockThreadExtentsMax(subVecEnd<dim::DimT<TWorkDiv>>(devProps.m_vuiBlockThreadExtentsMax));
            auto const uiBlockThreadCountMax(devProps.m_uiBlockThreadsCountMax);

            if(uiBlockThreadCountMax < vuiBlockThreadExtents.prod())
            {
                return false;
            }

            for(std::size_t i(0); i<dim::DimT<TWorkDiv>::value; ++i)
            {
                if((vuiGridBlockExtents[i] == 0)
                    || (vuiBlockThreadExtents[i] == 0)
                    || (vuiBlockThreadExtentsMax[i] < vuiBlockThreadExtents[i]))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
