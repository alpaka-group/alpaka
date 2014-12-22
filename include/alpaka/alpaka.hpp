/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_SERIAL_ENABLED
    #include <alpaka/serial/AccSerial.hpp>
#endif
#ifdef ALPAKA_THREADS_ENABLED
    #include <alpaka/threads/AccThreads.hpp>
#endif
#ifdef ALPAKA_FIBERS_ENABLED
    #include <alpaka/fibers/AccFibers.hpp>
#endif
#ifdef ALPAKA_OPENMP_ENABLED
    #include <alpaka/openmp/AccOpenMp.hpp>
#endif
#ifdef ALPAKA_CUDA_ENABLED
    #include <alpaka/cuda/AccCuda.hpp>
#endif

#include <alpaka/interfaces/Event.hpp>
#include <alpaka/interfaces/IAcc.hpp>
#include <alpaka/interfaces/KernelExecCreator.hpp>
#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/Memory.hpp>

#include <alpaka/host/WorkExtent.hpp>       // alpaka::WorkExtentHost

#include <iostream>                         // std::cout
#include <algorithm>                        // std::min

#include <boost/mpl/vector.hpp>             // boost::mpl::vector
#include <boost/mpl/filter_view.hpp>        // boost::mpl::filter_view
#include <boost/type_traits/is_same.hpp>    // boost::is_same
#include <boost/mpl/not.hpp>                // boost::not_
#include <boost/mpl/for_each.hpp>           // boost::mpl::for_each

//-----------------------------------------------------------------------------
//! The alpaka library namespace.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Logs the enabled accelerators.
    //-----------------------------------------------------------------------------
    static void logEnabledAccelerators()
    {
        std::cout << "Accelerators enabled: ";
#ifdef ALPAKA_SERIAL_ENABLED
        std::cout << "ALPAKA_SERIAL_ENABLED ";
#endif
#ifdef ALPAKA_THREADS_ENABLED
        std::cout << "ALPAKA_THREADS_ENABLED ";
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        std::cout << "ALPAKA_FIBERS_ENABLED ";
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        std::cout << "ALPAKA_OPENMP_ENABLED ";
#endif
#ifdef ALPAKA_CUDA_ENABLED
        std::cout << "ALPAKA_CUDA_ENABLED ";
#endif
        std::cout << std::endl;
    }

    //-----------------------------------------------------------------------------
    //! The detail namespace is used to separate implementation details from user accessible code.
    //-----------------------------------------------------------------------------
    namespace detail
    {
#ifdef ALPAKA_SERIAL_ENABLED
        using AccSerialIfAvailableElseVoid = AccSerial;
#else
        using AccSerialIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_THREADS_ENABLED
        using AccThreadsIfAvailableElseVoid = AccThreads;
#else
        using AccThreadsIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        using AccFibersIfAvailableElseVoid = AccFibers;
#else
        using AccFibersIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        using AccOpenMpIfAvailableElseVoid = AccOpenMp;
#else
        using AccOpenMpIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_CUDA_ENABLED
        using AccCudaIfAvailableElseVoid = AccCuda;
#else
        using AccCudaIfAvailableElseVoid = void;
#endif
        //-----------------------------------------------------------------------------
        //! A vector containing all available accelerators and void's.
        //-----------------------------------------------------------------------------
        using EnabledAcceleratorsVoid = 
            boost::mpl::vector<
                AccSerialIfAvailableElseVoid, 
                AccThreadsIfAvailableElseVoid,
                AccFibersIfAvailableElseVoid,
                AccOpenMpIfAvailableElseVoid,
                AccCudaIfAvailableElseVoid
            >;
    }
    //-----------------------------------------------------------------------------
    //! A vector containing all available accelerators.
    //-----------------------------------------------------------------------------
    using EnabledAccelerators = 
        boost::mpl::filter_view<
            detail::EnabledAcceleratorsVoid, 
            boost::mpl::not_<
                boost::is_same<
                    boost::mpl::_1, 
                    void
                >
            >
        >::type;

    namespace detail
    {
        struct CorrectMaxBlockKernelExtent
        {
            //-----------------------------------------------------------------------------
            //! \return The maximum block size per dimension supported by all of the given accelerators.
            //-----------------------------------------------------------------------------
            template<typename TAcc>
            void operator()(TAcc, alpaka::vec<3u> & v3uiBlockKernelExtent)
            {
                using TDeviceManager = alpaka::device::DeviceManager<TAcc>;
                auto const deviceProperties(TDeviceManager::getCurrentDevice().getProperties());
                auto const & v3uiBlockKernelsExtentMax(deviceProperties.m_v3uiBlockKernelsExtentMax);

                v3uiBlockKernelExtent = alpaka::vec<3u>(
                    std::min(v3uiBlockKernelExtent[0u], v3uiBlockKernelsExtentMax[0u]),
                    std::min(v3uiBlockKernelExtent[1u], v3uiBlockKernelsExtentMax[1u]),
                    std::min(v3uiBlockKernelExtent[2u], v3uiBlockKernelsExtentMax[2u])
                );
            }
        };

        struct CorrectMaxBlockKernelCount
        {
            //-----------------------------------------------------------------------------
            //! \return The maximum block size supported by all of the given accelerators.
            //-----------------------------------------------------------------------------
            template<typename TAcc>
            void operator()(TAcc, std::size_t & uiBlockKernelCount)
            {
                using TDeviceManager = alpaka::device::DeviceManager<TAcc>;
                auto const deviceProperties(TDeviceManager::getCurrentDevice().getProperties());
                auto const & uiBlockKernelCountMax(deviceProperties.m_uiBlockKernelsCountMax);

                uiBlockKernelCount = std::min(uiBlockKernelCount, uiBlockKernelCountMax);
            }
        };
    }

    //-----------------------------------------------------------------------------
    //! \return The maximum block size per dimension supported by all of the enabled accelerators.
    //-----------------------------------------------------------------------------
    alpaka::vec<3u> getMaxBlockKernelExtentEnabledAccelerators()
    {
        alpaka::vec<3u> v3uiMaxBlockKernelExtent(
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max(),
            std::numeric_limits<std::size_t>::max());

        boost::mpl::for_each<alpaka::EnabledAccelerators>(
            std::bind(detail::CorrectMaxBlockKernelExtent(), std::placeholders::_1, std::ref(v3uiMaxBlockKernelExtent))
        );

        return v3uiMaxBlockKernelExtent;
    }

    //-----------------------------------------------------------------------------
    //! \return The maximum block size supported by all of the enabled accelerators.
    //-----------------------------------------------------------------------------
    std::size_t getMaxBlockKernelCountEnabledAccelerators()
    {
        std::size_t uiMaxBlockKernelCount(
            std::numeric_limits<std::size_t>::max());

        boost::mpl::for_each<alpaka::EnabledAccelerators>(
            std::bind(detail::CorrectMaxBlockKernelCount(), std::placeholders::_1, std::ref(uiMaxBlockKernelCount))
        );

        return uiMaxBlockKernelCount;
    }

    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! \param uiMaxDivisor The maximum divisor.
        //! \param uiDividend The dividend.
        //! \return A number that satisfies the following conditions:
        //!     1) uiDividend/ret==0
        //!     2) ret<=uiMaxDivisor
        //-----------------------------------------------------------------------------
        std::size_t nextLowerOrEqualFactor(std::size_t const & uiMaxDivisor, std::size_t const & uiDividend)
        {
            std::size_t uiDivisor(uiMaxDivisor);
            // \TODO: This is not very efficient. Replace with a better algorithm.
            while((uiDividend%uiDivisor)!=0)
            {
                --uiDivisor;
            }
            return uiDivisor;
        }
    }

    //-----------------------------------------------------------------------------
    //! \param v3uiGridKernelsExtent        
    //!     The maximum divisor.
    //! \param bAdaptiveBlockKernelsExtent  
    //!     If the block kernels extent should be selected adaptively to the given accelerator
    //!     or the minimum supported by all accelerator.
    //! \return The work extent.
    // \TODO: Make this a template depending on Accelerator and Kernel
    //-----------------------------------------------------------------------------
    template<typename TAcc>
    alpaka::WorkExtent getValidWorkExtent(alpaka::vec<3u> const & v3uiGridKernelsExtent, bool const & bAdaptiveBlockKernelsExtent)
    {
        // \TODO: Print a warning when the grid kernels extent is a prime number and the resulting block kernels extent is 1.

        assert(v3uiGridKernelsExtent[0u]>0);
        assert(v3uiGridKernelsExtent[1u]>0);
        assert(v3uiGridKernelsExtent[2u]>0);

        alpaka::vec<3u> v3uiMaxBlockKernelsExtent;
        std::size_t uiMaxBlockKernelsCount;

        // Get the maximum block kernels extent depending on the the input.
        if(bAdaptiveBlockKernelsExtent)
        {
            using TDeviceManager = alpaka::device::DeviceManager<TAcc>;
            auto const deviceProperties(TDeviceManager::getCurrentDevice().getProperties());
            v3uiMaxBlockKernelsExtent = deviceProperties.m_v3uiBlockKernelsExtentMax;
            uiMaxBlockKernelsCount = deviceProperties.m_uiBlockKernelsCountMax;
        }
        else
        {
            v3uiMaxBlockKernelsExtent = alpaka::getMaxBlockKernelExtentEnabledAccelerators();
            uiMaxBlockKernelsCount = alpaka::getMaxBlockKernelCountEnabledAccelerators();
        }

        // Restrict the max block kernels extent with the grid kernels extent.
        // This removes dimensions not required.
        // This has to be done before the uiMaxBlockKernelsCount clipping to get the maximum correctly.
        v3uiMaxBlockKernelsExtent = alpaka::vec<3u>(
            std::min(v3uiMaxBlockKernelsExtent[0u], v3uiGridKernelsExtent[0u]),
            std::min(v3uiMaxBlockKernelsExtent[1u], v3uiGridKernelsExtent[1u]),
            std::min(v3uiMaxBlockKernelsExtent[2u], v3uiGridKernelsExtent[2u]));

        // If the block kernels extent allows more kernels then available on the accelerator, clip it.
        std::size_t const uiBlockKernelsCount(v3uiMaxBlockKernelsExtent.prod());
        if(uiBlockKernelsCount>uiMaxBlockKernelsCount)
        {
            // Very primitive clipping. Just halve it until it fits.
            // \TODO: Use a better algorithm for clipping.
            while(v3uiMaxBlockKernelsExtent.prod()>uiMaxBlockKernelsCount)
            {
                v3uiMaxBlockKernelsExtent = alpaka::vec<3u>(
                    std::max(static_cast<std::size_t>(1u), static_cast<std::size_t>(v3uiMaxBlockKernelsExtent[0u]/2u)),
                    std::max(static_cast<std::size_t>(1u), static_cast<std::size_t>(v3uiMaxBlockKernelsExtent[1u]/2u)),
                    std::max(static_cast<std::size_t>(1u), static_cast<std::size_t>(v3uiMaxBlockKernelsExtent[2u]/2u)));
            }
        }

        // Make the block kernels extent divide the grid kernels extent.
        alpaka::vec<3u> const v3uiBlockKernelsExtent(
            detail::nextLowerOrEqualFactor(v3uiMaxBlockKernelsExtent[0u], v3uiGridKernelsExtent[0u]),
            detail::nextLowerOrEqualFactor(v3uiMaxBlockKernelsExtent[1u], v3uiGridKernelsExtent[1u]),
            detail::nextLowerOrEqualFactor(v3uiMaxBlockKernelsExtent[2u], v3uiGridKernelsExtent[2u]));

        // Set the grid blocks extent.
        alpaka::vec<3u> const v3uiGridBlocksExtent(
            v3uiGridKernelsExtent[0u]/v3uiBlockKernelsExtent[0u],
            v3uiGridKernelsExtent[1u]/v3uiBlockKernelsExtent[1u],
            v3uiGridKernelsExtent[2u]/v3uiBlockKernelsExtent[2u]);

        return alpaka::WorkExtent(v3uiGridBlocksExtent, v3uiBlockKernelsExtent);
    }
}
