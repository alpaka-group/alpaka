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

#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/dev/Traits.hpp>                // dev::DevManT
#include <alpaka/acc/Traits.hpp>                // getAccDevProps

#include <alpaka/core/ForEachType.hpp>          // forEachType
#include <alpaka/core/Vec.hpp>                  // Vec
#include <alpaka/core/Common.hpp>               // ALPAKA_FN_HOST

#include <cmath>                                // std::ceil
#include <algorithm>                            // std::min
#include <functional>                           // std::bind
#include <set>                                  // std::set
#include <array>                                // std::array

//-----------------------------------------------------------------------------
//! The alpaka library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The block extent subdivision restrictions.
        //#############################################################################
        enum class BlockExtentsSubDivRestrictions
        {
            EqualExtents,       //!< The block thread extents will be equal in all dimensions.
            CloseToEqualExtents,//!< The block thread extents will be as close to equal as possible in all dimensions.
            Unrestricted,
        };

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
            ALPAKA_FN_HOST auto nextDivisorLowerOrEqual(
                T const & uiMaxDivisor,
                T const & uiDividend)
            -> T
            {
                T uiDivisor(uiMaxDivisor);

                while((uiDividend%uiDivisor) != 0)
                {
                    --uiDivisor;
                }

                return uiDivisor;
            }
            //-----------------------------------------------------------------------------
            //! \param uiVal The value to find divisors of.
            //! \param uiMaxDivisor The maximum.
            //! \return A list of all divisors less then or euqal the given maximum.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                typename = typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            std::set<T> allDivisorsLessOrEqual(
                T const & uiVal,
                T const & uiMaxDivisor)
            {
                std::set<T> setDivisors;

                for(T i(1); i <= std::min(uiVal, uiMaxDivisor); ++i)
                {
                    if(uiVal % i == 0)
                    {
                        setDivisors.insert(uiVal/i);
                    }
                }

                return setDivisors;
            }
        }

        //-----------------------------------------------------------------------------
        //! Subdivides the given grid thread extents into blocks restricted by:
        //! 1. The maximum block thread extents
        //! 2. The maximum block thread count
        //! 3. The requirement of the block thread extents to divide the grid thread extents without remainder
        //! 4. The requirement of the block extents.
        //!
        //! \param vuiGridThreadExtents
        //!     The full extents of threads in the grid.
        //! \param vuiMaxBlockThreadExtents
        //!     The maximum extents of threads in a block.
        //! \param uiMaxBlockThreadsCount
        //!     The maximum number of threads in a block.
        //! \param bRequireBlockThreadExtentsToDivideGridThreadExtents
        //!     If this is true, the grid thread extents will be multiples of the corresponding block thread extents.
        //!     NOTE: If this is true and vuiGridThreadExtents is prime (or otherwise bad chosen) in a dimension, the block thread extent will be one in this dimension.
        //! \param eBlockExtentsSubDivRestrictions
        //!     The block extent subdivision restrictions.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto subDivideGridThreads(
            Vec<TDim, TSize> const & vuiGridThreadExtents,
            Vec<TDim, TSize> const & vuiMaxBlockThreadExtents,
            TSize const & uiMaxBlockThreadsCount,
            bool bRequireBlockThreadExtentsToDivideGridThreadExtents = true,
            BlockExtentsSubDivRestrictions eBlockExtentsSubDivRestrictions = BlockExtentsSubDivRestrictions::Unrestricted)
        -> workdiv::WorkDivMembers<TDim, TSize>
        {
            // Assert valid input.
            assert(uiMaxBlockThreadsCount>0u);
            for(typename TDim::value_type i(0u); i<TDim::value; ++i)
            {
                assert(vuiGridThreadExtents[i]>0u);
                assert(vuiMaxBlockThreadExtents[i]>0u);
            }

            // Initialize the block thread extents with the maximum possible.
            auto vuiBlockThreadExtents(vuiMaxBlockThreadExtents);

            // Restrict the max block thread extents with the grid thread extents.
            // This removes dimensions not required in the given grid thread extents.
            // This has to be done before the uiMaxBlockThreadsCount clipping to get the maximum correctly.
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                vuiBlockThreadExtents[i] = std::min(vuiBlockThreadExtents[i], vuiGridThreadExtents[i]);
            }

            // For equal block thread extents, restrict it to its minimum component.
            // For example (512, 256, 1024) will get (256, 256, 256).
            if(eBlockExtentsSubDivRestrictions == BlockExtentsSubDivRestrictions::EqualExtents)
            {
                auto const uiMinBlockThreadExtent(vuiBlockThreadExtents.min());
                for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                {
                    vuiBlockThreadExtents[i] = uiMinBlockThreadExtent;
                }
            }

            // Adjust vuiBlockThreadExtents if its product is too large.
            if(vuiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
            {
                // Satisfy the following equation:
                // uiMaxBlockThreadsCount >= vuiBlockThreadExtents.prod()
                // For example 1024 >= 512 * 512 * 1024

                // For equal block thread extent this is easily the nth root of uiMaxBlockThreadsCount.
                if(eBlockExtentsSubDivRestrictions == BlockExtentsSubDivRestrictions::EqualExtents)
                {
                    double const fNthRoot(std::pow(uiMaxBlockThreadsCount, 1.0/static_cast<double>(TDim::value)));
                    TSize const uiNthRoot(static_cast<TSize>(fNthRoot));
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        vuiBlockThreadExtents[i] = uiNthRoot;
                    }
                }
                else if(eBlockExtentsSubDivRestrictions == BlockExtentsSubDivRestrictions::CloseToEqualExtents)
                {
                    // Very primitive clipping. Just halve the largest value until it fits.
                    while(vuiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
                    {
                        auto const uiMaxElemIdx(vuiBlockThreadExtents.maxElem());
                        vuiBlockThreadExtents[uiMaxElemIdx] = vuiBlockThreadExtents[uiMaxElemIdx] / 2u;
                    }
                }
                else
                {
                    // Very primitive clipping. Just halve the smallest value until it fits.
                    while(vuiBlockThreadExtents.prod() > uiMaxBlockThreadsCount)
                    {
                        // Compute the minimum element index but ignore ones.
                        // Ones compare always larger to everything else.
                        auto const uiMinElemIdx(
                            static_cast<TSize>(
                                std::distance(
                                    &vuiBlockThreadExtents[0],
                                    std::min_element(
                                        &vuiBlockThreadExtents[0],
                                        &vuiBlockThreadExtents[TDim::value-1u],
                                        [](TSize const & a, TSize const & b)
                                        {
                                            // This first case is redundant.
                                            /*if((a == 1u) && (b == 1u))
                                            {
                                                return false;
                                            }
                                            else */if(a == 1u)
                                            {
                                                return false;
                                            }
                                            else if(b == 1u)
                                            {
                                                return true;
                                            }
                                            else
                                            {
                                                return a < b;
                                            }
                                        }))));
                        vuiBlockThreadExtents[uiMinElemIdx] = vuiBlockThreadExtents[uiMinElemIdx] / 2u;
                    }
                }
            }

            // Make the block thread extents divide the grid thread extents.
            if(bRequireBlockThreadExtentsToDivideGridThreadExtents)
            {
                if(eBlockExtentsSubDivRestrictions == BlockExtentsSubDivRestrictions::EqualExtents)
                {
                    // For equal size block extents we have to compute the gcd of all grid thread extents that is less then the current maximal block thread extent.
                    // For this we compute the divisors of all grid thread extents less then the current maximal block thread extent.
                    std::array<std::set<TSize>, TDim::value> gridThreadExtentsDivisors;
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        gridThreadExtentsDivisors[i] =
                            detail::allDivisorsLessOrEqual(
                                vuiGridThreadExtents[i],
                                vuiBlockThreadExtents[i]);
                    }
                    // The maximal common divisor of all block thread extents is the optimal solution.
                    std::set<TSize> intersects[2u];
                    for(typename TDim::value_type i(1u); i<TDim::value; ++i)
                    {
                        intersects[(i-1u)%2u] = gridThreadExtentsDivisors[0];
                        intersects[(i)%2u].clear();
                        set_intersection(
                            intersects[(i-1u)%2u].begin(),
                            intersects[(i-1u)%2u].end(),
                            gridThreadExtentsDivisors[i].begin(),
                            gridThreadExtentsDivisors[i].end(),
                            std::inserter(intersects[i%2], intersects[i%2u].begin()));
                    }
                    TSize const uiMaxCommonDivisor(*(--intersects[(TDim::value-1)%2u].end()));
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        vuiBlockThreadExtents[i] = uiMaxCommonDivisor;
                    }
                }
                else if(eBlockExtentsSubDivRestrictions == BlockExtentsSubDivRestrictions::CloseToEqualExtents)
                {
                    for(typename TDim::value_type i(0); i<TDim::value; ++i)
                    {
                        vuiBlockThreadExtents[i] =
                            detail::nextDivisorLowerOrEqual(
                                vuiBlockThreadExtents[i],
                                vuiGridThreadExtents[i]);
                    }
                }
                else
                {
                    for(typename TDim::value_type i(0); i<TDim::value; ++i)
                    {
                        vuiBlockThreadExtents[i] =
                            detail::nextDivisorLowerOrEqual(
                                vuiBlockThreadExtents[i],
                                vuiGridThreadExtents[i]);
                    }
                }
            }

            // Set the grid block extents (rounded to the next integer not less then the quotient.
            auto vuiGridBlockExtents(Vec<TDim, TSize>::ones());
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                vuiGridBlockExtents[i] =
                    static_cast<TSize>(
                        std::ceil(static_cast<double>(vuiGridThreadExtents[i])
                        / static_cast<double>(vuiBlockThreadExtents[i])));
            }

            return workdiv::WorkDivMembers<TDim, TSize>(vuiGridBlockExtents, vuiBlockThreadExtents);
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAccSeq The accelerator sequence for which this work division has to be valid.
        //! \param gridThreadExtents The full extents of threads in the grid.
        //! \param bRequireBlockThreadExtentsToDivideGridThreadExtents If the grid thread extents have to be a multiple of the block thread extents.
        //! \return The work division.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TExtents,
            typename TDev>
        ALPAKA_FN_HOST auto getValidWorkDiv(
            TDev const & dev,
            TExtents const & gridThreadExtents = TExtents(),
            bool bRequireBlockThreadExtentsToDivideGridThreadExtents = true,
            BlockExtentsSubDivRestrictions eBlockExtentsSubDivRestrictions = BlockExtentsSubDivRestrictions::Unrestricted)
        -> workdiv::WorkDivMembers<dim::DimT<TExtents>, size::SizeT<TAcc>>
        {
            static_assert(
                dim::DimT<TExtents>::value == dim::DimT<TAcc>::value,
                "The dimension of TAcc and the dimension of TExtents have to be identical!");
            static_assert(
                std::is_same<size::SizeT<TExtents>, size::SizeT<TAcc>>::value,
                "The size type of TAcc and the size type of TExtents have to be identical!");

            auto const devProps(acc::getAccDevProps<TAcc>(dev));

            return subDivideGridThreads(
                extent::getExtentsVec(gridThreadExtents),
                devProps.m_vuiBlockThreadExtentsMax,
                devProps.m_uiBlockThreadsCountMax,
                bRequireBlockThreadExtentsToDivideGridThreadExtents,
                eBlockExtentsSubDivRestrictions);
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
        ALPAKA_FN_HOST auto isValidWorkDiv(
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

            for(typename dim::DimT<TWorkDiv>::value_type i(0); i<dim::DimT<TWorkDiv>::value; ++i)
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
