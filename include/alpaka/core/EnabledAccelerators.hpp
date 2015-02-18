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

#include <alpaka/traits/Acc.hpp>            // traits::GetAccName

#include <alpaka/core/WorkDivHelpers.hpp>   // getMaxBlockKernelExtentsAccelerators

#include <boost/mpl/vector.hpp>             // boost::mpl::vector
#include <boost/mpl/filter_view.hpp>        // boost::mpl::filter_view
#include <boost/mpl/not.hpp>                // boost::not_
#include <boost/type_traits/is_same.hpp>    // boost::is_same
#include <boost/mpl/for_each.hpp>           // boost::mpl::for_each

#include <iosfwd>                           // std::ostream

//-----------------------------------------------------------------------------
//! The alpaka library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The accelerators.
    //-----------------------------------------------------------------------------
    namespace acc
    {
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
            //#############################################################################
            //! A vector containing all available accelerators and void's.
            //#############################################################################
            using EnabledAcceleratorsVoid = 
                boost::mpl::vector<
                    AccSerialIfAvailableElseVoid, 
                    AccThreadsIfAvailableElseVoid,
                    AccFibersIfAvailableElseVoid,
                    AccOpenMpIfAvailableElseVoid,
                    AccCudaIfAvailableElseVoid
                >;
        }

        //#############################################################################
        //! A vector containing all available accelerators.
        //#############################################################################
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
            //#############################################################################
            //! The accelerator name write wrapper.
            //#############################################################################
            struct GetAccName
            {
                template<
                    typename TAcc> 
                ALPAKA_FCT_HOST void operator()(TAcc &, std::ostream & os)
                {
                    os << acc::getAccName<TAcc>();
                    os << " ";
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! Writes the enabled accelerators to the given stream.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST void writeEnabledAccelerators(std::ostream & os)
        {
            os << "Accelerators enabled: ";

            boost::mpl::for_each<EnabledAccelerators>(
                std::bind(detail::GetAccName(), std::placeholders::_1, std::ref(os))
                );

            os << std::endl;
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block kernels extents supported by all of the enabled accelerators.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST Vec<3u> getMaxBlockKernelExtentsEnabledAccelerators()
        {
            return workdiv::getMaxBlockKernelExtentsAccelerators<acc::EnabledAccelerators>();
        }

        //-----------------------------------------------------------------------------
        //! \return The maximum block kernels count supported by all of the enabled accelerators.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST UInt getMaxBlockKernelCountEnabledAccelerators()
        {
            return workdiv::getMaxBlockKernelCountAccelerators<acc::EnabledAccelerators>();
        }
    }
}