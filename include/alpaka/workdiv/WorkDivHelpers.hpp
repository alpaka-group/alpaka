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

#include <alpaka/dev/Traits.hpp>                // dev::DevMan
#include <alpaka/acc/Traits.hpp>                // getAccDevProps

#include <alpaka/vec/Vec.hpp>                   // Vec
#include <alpaka/core/Common.hpp>               // ALPAKA_FN_HOST

#include <cassert>                              // assert
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
        //! The grid block extents subdivision restrictions.
        //#############################################################################
        enum class GridBlockExtentsSubDivRestrictions
        {
            EqualExtents,       //!< The block thread extents will be equal in all dimensions.
            CloseToEqualExtents,//!< The block thread extents will be as close to equal as possible in all dimensions.
            Unrestricted,
        };

        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! \param maxDivisor The maximum divisor.
            //! \param dividend The dividend.
            //! \return The biggest number that satisfies the following conditions:
            //!     1) dividend/ret==0
            //!     2) ret<=maxDivisor
            //-----------------------------------------------------------------------------
            template<
                typename T,
                typename = typename std::enable_if<std::is_integral<T>::value>::type>
            ALPAKA_FN_HOST auto nextDivisorLowerOrEqual(
                T const & maxDivisor,
                T const & dividend)
            -> T
            {
                T divisor(maxDivisor);

                core::assertValueUnsigned(dividend);
                core::assertValueUnsigned(maxDivisor);
                assert(dividend <= maxDivisor);

                while((dividend%divisor) != 0)
                {
                    --divisor;
                }

                return divisor;
            }
            //-----------------------------------------------------------------------------
            //! \param val The value to find divisors of.
            //! \param maxDivisor The maximum.
            //! \return A list of all divisors less then or equal to the given maximum.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                typename = typename std::enable_if<std::is_integral<T>::value>::type>
            ALPAKA_FN_HOST auto allDivisorsLessOrEqual(
                T const & val,
                T const & maxDivisor)
            -> std::set<T>
            {
                std::set<T> divisorSet;

                core::assertValueUnsigned(val);
                core::assertValueUnsigned(maxDivisor);
                assert(maxDivisor <= val);

                for(T i(1); i <= std::min(val, maxDivisor); ++i)
                {
                    if(val % i == 0)
                    {
                        divisorSet.insert(val/i);
                    }
                }

                return divisorSet;
            }
        }

        //-----------------------------------------------------------------------------
        //! Subdivides the given grid thread extents into blocks restricted by:
        //! 1. The maximum block thread extents
        //! 2. The maximum block thread count
        //! 3. The requirement of the block thread extents to divide the grid thread extents without remainder
        //! 4. The requirement of the block extents.
        //!
        //! \param gridThreadExtents
        //!     The full extents of threads in the grid.
        //! \param blockThreadExtentsMax
        //!     The maximum extents of threads in a block.
        //! \param blockThreadsCountMax
        //!     The maximum number of threads in a block.
        //! \param requireBlockThreadExtentsToDivideGridThreadExtents
        //!     If this is true, the grid thread extents will be multiples of the corresponding block thread extents.
        //!     NOTE: If this is true and gridThreadExtents is prime (or otherwise bad chosen) in a dimension, the block thread extent will be one in this dimension.
        //! \param gridBlockExtentsSubDivRestrictions
        //!     The grid block extent subdivision restrictions.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto subDivideGridThreads(
            Vec<TDim, TSize> const & gridThreadExtents,
            Vec<TDim, TSize> const & blockThreadExtentsMax,
            TSize const & blockThreadsCountMax,
            bool requireBlockThreadExtentsToDivideGridThreadExtents = true,
            GridBlockExtentsSubDivRestrictions gridBlockExtentsSubDivRestrictions = GridBlockExtentsSubDivRestrictions::Unrestricted)
        -> workdiv::WorkDivMembers<TDim, TSize>
        {
            // Assert valid input.
            assert(blockThreadsCountMax>0u);
            for(typename TDim::value_type i(0u); i<TDim::value; ++i)
            {
                assert(gridThreadExtents[i]>0u);
                assert(blockThreadExtentsMax[i]>0u);
            }

            // Initialize the block thread extents with the maximum possible.
            auto blockThreadExtents(blockThreadExtentsMax);

            // Restrict the max block thread extents with the grid thread extents.
            // This removes dimensions not required in the given grid thread extents.
            // This has to be done before the blockThreadsCountMax clipping to get the maximum correctly.
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                blockThreadExtents[i] = std::min(blockThreadExtents[i], gridThreadExtents[i]);
            }

            // For equal block thread extents, restrict it to its minimum component.
            // For example (512, 256, 1024) will get (256, 256, 256).
            if(gridBlockExtentsSubDivRestrictions == GridBlockExtentsSubDivRestrictions::EqualExtents)
            {
                auto const minBlockThreadExtent(blockThreadExtents.min());
                for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                {
                    blockThreadExtents[i] = minBlockThreadExtent;
                }
            }

            // Adjust blockThreadExtents if its product is too large.
            if(blockThreadExtents.prod() > blockThreadsCountMax)
            {
                // Satisfy the following equation:
                // blockThreadsCountMax >= blockThreadExtents.prod()
                // For example 1024 >= 512 * 512 * 1024

                // For equal block thread extent this is easily the nth root of blockThreadsCountMax.
                if(gridBlockExtentsSubDivRestrictions == GridBlockExtentsSubDivRestrictions::EqualExtents)
                {
                    double const fNthRoot(std::pow(blockThreadsCountMax, 1.0/static_cast<double>(TDim::value)));
                    TSize const nthRoot(static_cast<TSize>(fNthRoot));
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        blockThreadExtents[i] = nthRoot;
                    }
                }
                else if(gridBlockExtentsSubDivRestrictions == GridBlockExtentsSubDivRestrictions::CloseToEqualExtents)
                {
                    // Very primitive clipping. Just halve the largest value until it fits.
                    while(blockThreadExtents.prod() > blockThreadsCountMax)
                    {
                        auto const maxElemIdx(blockThreadExtents.maxElem());
                        blockThreadExtents[maxElemIdx] = blockThreadExtents[maxElemIdx] / 2u;
                    }
                }
                else
                {
                    // Very primitive clipping. Just halve the smallest value until it fits.
                    while(blockThreadExtents.prod() > blockThreadsCountMax)
                    {
                        // Compute the minimum element index but ignore ones.
                        // Ones compare always larger to everything else.
                        auto const minElemIdx(
                            static_cast<TSize>(
                                std::distance(
                                    &blockThreadExtents[0],
                                    std::min_element(
                                        &blockThreadExtents[0],
                                        &blockThreadExtents[TDim::value-1u],
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
                        blockThreadExtents[minElemIdx] = blockThreadExtents[minElemIdx] / 2u;
                    }
                }
            }

            // Make the block thread extents divide the grid thread extents.
            if(requireBlockThreadExtentsToDivideGridThreadExtents)
            {
                if(gridBlockExtentsSubDivRestrictions == GridBlockExtentsSubDivRestrictions::EqualExtents)
                {
                    // For equal size block extents we have to compute the gcd of all grid thread extents that is less then the current maximal block thread extent.
                    // For this we compute the divisors of all grid thread extents less then the current maximal block thread extent.
                    std::array<std::set<TSize>, TDim::value> gridThreadExtentsDivisors;
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        gridThreadExtentsDivisors[i] =
                            detail::allDivisorsLessOrEqual(
                                gridThreadExtents[i],
                                blockThreadExtents[i]);
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
                    TSize const maxCommonDivisor(*(--intersects[(TDim::value-1)%2u].end()));
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        blockThreadExtents[i] = maxCommonDivisor;
                    }
                }
                else if(gridBlockExtentsSubDivRestrictions == GridBlockExtentsSubDivRestrictions::CloseToEqualExtents)
                {
                    for(typename TDim::value_type i(0); i<TDim::value; ++i)
                    {
                        blockThreadExtents[i] =
                            detail::nextDivisorLowerOrEqual(
                                blockThreadExtents[i],
                                gridThreadExtents[i]);
                    }
                }
                else
                {
                    for(typename TDim::value_type i(0); i<TDim::value; ++i)
                    {
                        blockThreadExtents[i] =
                            detail::nextDivisorLowerOrEqual(
                                blockThreadExtents[i],
                                gridThreadExtents[i]);
                    }
                }
            }

            // Set the grid block extents (rounded to the next integer not less then the quotient.
            auto gridBlockExtents(Vec<TDim, TSize>::ones());
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                gridBlockExtents[i] =
                    static_cast<TSize>(
                        std::ceil(static_cast<double>(gridThreadExtents[i])
                        / static_cast<double>(blockThreadExtents[i])));
            }

            return workdiv::WorkDivMembers<TDim, TSize>(gridBlockExtents, blockThreadExtents);
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAcc The accelerator for which this work division has to be valid.
        //! \param dev The device for which this work division has to be valid.
        //! \param gridThreadExtents The full extents of threads in the grid.
        //! \param requireBlockThreadExtentsToDivideGridThreadExtents If the grid thread extents have to be a multiple of the block thread extents.
        //! \param gridBlockExtentsSubDivRestrictions The grid block extent subdivision restrictions.
        //! \return The work division.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename TExtents,
            typename TDev>
        ALPAKA_FN_HOST auto getValidWorkDiv(
            TDev const & dev,
            TExtents const & gridThreadExtents = TExtents(),
            bool requireBlockThreadExtentsToDivideGridThreadExtents = true,
            GridBlockExtentsSubDivRestrictions gridBlockExtentsSubDivRestrictions = GridBlockExtentsSubDivRestrictions::Unrestricted)
        -> workdiv::WorkDivMembers<dim::Dim<TExtents>, size::Size<TAcc>>
        {
            static_assert(
                dim::Dim<TExtents>::value == dim::Dim<TAcc>::value,
                "The dimension of TAcc and the dimension of TExtents have to be identical!");
            static_assert(
                std::is_same<size::Size<TExtents>, size::Size<TAcc>>::value,
                "The size type of TAcc and the size type of TExtents have to be identical!");

            auto const devProps(acc::getAccDevProps<TAcc>(dev));

            return subDivideGridThreads(
                extent::getExtentsVec(gridThreadExtents),
                devProps.m_blockThreadExtentsMax,
                devProps.m_blockThreadsCountMax,
                requireBlockThreadExtentsToDivideGridThreadExtents,
                gridBlockExtentsSubDivRestrictions);
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
            auto const gridBlockExtents(getWorkDiv<Grid, Blocks>(workDiv));
            auto const blockThreadExtents(getWorkDiv<Block, Threads>(workDiv));

            auto const devProps(acc::getAccDevProps<TAcc>(dev));
            auto const blockThreadExtentsMax(vec::subVecEnd<dim::Dim<TWorkDiv>>(devProps.m_blockThreadExtentsMax));
            auto const blockThreadCountMax(devProps.m_blockThreadsCountMax);

            if(blockThreadCountMax < blockThreadExtents.prod())
            {
                return false;
            }

            for(typename dim::Dim<TWorkDiv>::value_type i(0); i<dim::Dim<TWorkDiv>::value; ++i)
            {
                if((gridBlockExtents[i] == 0)
                    || (blockThreadExtents[i] == 0)
                    || (blockThreadExtentsMax[i] < blockThreadExtents[i]))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
