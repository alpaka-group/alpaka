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

#include <alpaka/core/Vec.hpp>                  // alpaka::Vec
#include <alpaka/core/EnabledAccelerators.hpp>  // alpaka::acc::EnabledAccelerators

#include <alpaka/traits/Device.hpp>             // alpaka::dev::GetDevManT

#include <algorithm>                            // std::min

//-----------------------------------------------------------------------------
//! The alpaka library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The maximum block size per dimension correction wrapper.
        //#############################################################################
        struct CorrectMaxBlockKernelExtents
        {
            //-----------------------------------------------------------------------------
            //! Corrects the maximum block size per dimension if it is larger then the one supported by the given accelerator type.
            //-----------------------------------------------------------------------------
            template<
                typename TAcc>
            void operator()(
                TAcc, 
                alpaka::Vec<3u> & v3uiBlockKernelExtents)
            {
                auto const devProps(alpaka::dev::getDevProps(alpaka::dev::GetDevManT<TAcc>::getCurrentDevice()));
                auto const & v3uiBlockKernelsExtentsMax(devProps.m_v3uiBlockKernelsExtentsMax);

                v3uiBlockKernelExtents = alpaka::Vec<3u>(
                    std::min(v3uiBlockKernelExtents[0u], v3uiBlockKernelsExtentsMax[0u]),
                    std::min(v3uiBlockKernelExtents[1u], v3uiBlockKernelsExtentsMax[1u]),
                    std::min(v3uiBlockKernelExtents[2u], v3uiBlockKernelsExtentsMax[2u]));
            }
        };

        //#############################################################################
        //! The maximum block size correction wrapper.
        //#############################################################################
        struct CorrectMaxBlockKernelCount
        {
            //-----------------------------------------------------------------------------
            //! Corrects the maximum block size if it is larger then the one supported by the given accelerator type.
            //-----------------------------------------------------------------------------
            template<
                typename TAcc>
            void operator()(
                TAcc, 
                std::size_t & uiBlockKernelCount)
            {
                auto const devProps(alpaka::dev::getDevProps(alpaka::dev::GetDevManT<TAcc>::getCurrentDevice()));
                auto const & uiBlockKernelCountMax(devProps.m_uiBlockKernelsCountMax);

                uiBlockKernelCount = std::min(uiBlockKernelCount, uiBlockKernelCountMax);
            }
        };
    }

    //-----------------------------------------------------------------------------
    //! \return The maximum block size per dimension supported by all of the enabled accelerators.
    //-----------------------------------------------------------------------------
    alpaka::Vec<3u> getMaxBlockKernelExtentsEnabledAccelerators()
    {
        alpaka::Vec<3u> v3uiMaxBlockKernelExtents(
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max());

        boost::mpl::for_each<acc::EnabledAccelerators>(
            std::bind(detail::CorrectMaxBlockKernelExtents(), std::placeholders::_1, std::ref(v3uiMaxBlockKernelExtents))
            );

        return v3uiMaxBlockKernelExtents;
    }

    //-----------------------------------------------------------------------------
    //! \return The maximum block size supported by all of the enabled accelerators.
    //-----------------------------------------------------------------------------
    std::size_t getMaxBlockKernelCountEnabledAccelerators()
    {
        std::size_t uiMaxBlockKernelCount(
            std::numeric_limits<std::size_t>::max());

        boost::mpl::for_each<acc::EnabledAccelerators>(
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
        std::size_t nextLowerOrEqualFactor(
            std::size_t const & uiMaxDivisor, 
            std::size_t const & uiDividend)
        {
            std::size_t uiDivisor(uiMaxDivisor);
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
        //#############################################################################
        alpaka::WorkDiv subdivideGridKernels(
            alpaka::Vec<3u> const & v3uiGridKernelsExtents,
            alpaka::Vec<3u> const & v3uiMaxBlockKernelsExtents,
            std::size_t uiMaxBlockKernelsCount)
        {
            assert(v3uiGridKernelsExtents[0u]>0);
            assert(v3uiGridKernelsExtents[1u]>0);
            assert(v3uiGridKernelsExtents[2u]>0);

            // 1. Restrict the max block kernels extents with the grid kernels extents.
            // This removes dimensions not required in the given gird kernels extents.
            // This has to be done before the uiMaxBlockKernelsCount clipping to get the maximum correctly.
            alpaka::Vec<3u> v3uiBlockKernelsExtents(
                std::min(v3uiMaxBlockKernelsExtents[0u], v3uiGridKernelsExtents[0u]),
                std::min(v3uiMaxBlockKernelsExtents[1u], v3uiGridKernelsExtents[1u]),
                std::min(v3uiMaxBlockKernelsExtents[2u], v3uiGridKernelsExtents[2u]));

            // 2. If the block kernels extents require more kernels then available on the accelerator, clip it.
            if(v3uiBlockKernelsExtents.prod() > uiMaxBlockKernelsCount)
            {
                // Very primitive clipping. Just halve it until it fits.
                // \TODO: Use a better algorithm for clipping.
                while(v3uiBlockKernelsExtents.prod() > uiMaxBlockKernelsCount)
                {
                    v3uiBlockKernelsExtents = alpaka::Vec<3u>(
                        std::max(static_cast<std::size_t>(1u), static_cast<std::size_t>(v3uiBlockKernelsExtents[0u] / 2u)),
                        std::max(static_cast<std::size_t>(1u), static_cast<std::size_t>(v3uiBlockKernelsExtents[1u] / 2u)),
                        std::max(static_cast<std::size_t>(1u), static_cast<std::size_t>(v3uiBlockKernelsExtents[2u] / 2u)));
                }
            }

            // Make the block kernels extents divide the grid kernels extents.
            v3uiBlockKernelsExtents = alpaka::Vec<3u>(
                detail::nextLowerOrEqualFactor(v3uiBlockKernelsExtents[0u], v3uiGridKernelsExtents[0u]),
                detail::nextLowerOrEqualFactor(v3uiBlockKernelsExtents[1u], v3uiGridKernelsExtents[1u]),
                detail::nextLowerOrEqualFactor(v3uiBlockKernelsExtents[2u], v3uiGridKernelsExtents[2u]));

            // Set the grid blocks extents.
            alpaka::Vec<3u> const v3uiGridBlocksExtents(
                v3uiGridKernelsExtents[0u] / v3uiBlockKernelsExtents[0u],
                v3uiGridKernelsExtents[1u] / v3uiBlockKernelsExtents[1u],
                v3uiGridKernelsExtents[2u] / v3uiBlockKernelsExtents[2u]);

            return alpaka::WorkDiv(v3uiGridBlocksExtents, v3uiBlockKernelsExtents);
        }

        //#############################################################################
        //! The work division trait.
        // \TODO: Make this a template depending on Accelerator and Kernel
        //#############################################################################
        template<
            typename TAcc = void>
        struct GetValidWorkDiv
        {
            static alpaka::WorkDiv getValidWorkDiv(
                alpaka::Vec<3u> const & v3uiGridKernelsExtents)
            {
                // Get the maximum block kernels extents.
                auto const devProps(alpaka::dev::getDevProps(alpaka::dev::GetDevManT<TAcc>::getCurrentDevice()));

                return subdivideGridKernels(
                    v3uiGridKernelsExtents,
                    devProps.m_v3uiBlockKernelsExtentsMax,
                    devProps.m_uiBlockKernelsCountMax);
            }
        };


        //#############################################################################
        //! The work division trait specialization returning the minimum supported by all enabled accelerators.
        //#############################################################################
        template<>
        struct GetValidWorkDiv<
            void>
        {
            static alpaka::WorkDiv getValidWorkDiv(
                alpaka::Vec<3u> const & v3uiGridKernelsExtents)
            {
                return subdivideGridKernels(
                    v3uiGridKernelsExtents,
                    alpaka::getMaxBlockKernelExtentsEnabledAccelerators(),
                    alpaka::getMaxBlockKernelCountEnabledAccelerators());
            }
        };
    }

    //-----------------------------------------------------------------------------
    //! \tparam TAcc  
    //!     If an accelerator is given, the block kernels extents is selected adaptively to the given accelerator,
    //!     otherwise the minimum supported by all accelerators is calculated.
    //! \param v3uiGridKernelsExtents        
    //!     The full extents of kernels in the grid.
    //! \return The work division.
    //-----------------------------------------------------------------------------
    template<
        typename TAcc = void>
    alpaka::WorkDiv getValidWorkDiv(
        alpaka::Vec<3u> const & v3uiGridKernelsExtents)
    {
        return detail::GetValidWorkDiv<TAcc>::getValidWorkDiv(v3uiGridKernelsExtents);
    }
}
