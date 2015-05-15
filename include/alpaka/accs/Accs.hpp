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
    #include <alpaka/accs/serial/Serial.hpp>
#endif
#ifdef ALPAKA_THREADS_ENABLED
    #include <alpaka/accs/threads/Threads.hpp>
#endif
#ifdef ALPAKA_FIBERS_ENABLED
    #include <alpaka/accs/fibers/Fibers.hpp>
#endif
#ifdef ALPAKA_OPENMP2_ENABLED
    #include <alpaka/accs/omp/omp2/Omp2.hpp>
#endif
#if defined(ALPAKA_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/accs/cuda/Cuda.hpp>
#endif

#include <alpaka/traits/Acc.hpp>            // traits::GetAccName

#include <alpaka/core/ForEachType.hpp>      // forEachType
#include <alpaka/core/WorkDivHelpers.hpp>   // getMaxBlockThreadExtentsAccelerators

#include <boost/mpl/vector.hpp>             // boost::mpl::vector
#include <boost/mpl/filter_view.hpp>        // boost::mpl::filter_view
#include <boost/mpl/not.hpp>                // boost::not_
#include <boost/mpl/placeholders.hpp>       // boost::mpl::_1
#include <boost/type_traits/is_same.hpp>    // boost::is_same

#include <iosfwd>                           // std::ostream

//-----------------------------------------------------------------------------
//! The alpaka library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The accelerators.
    //-----------------------------------------------------------------------------
    namespace accs
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
#ifdef ALPAKA_OPENMP2_ENABLED
            using AccOmp2IfAvailableElseVoid = AccOmp2;
#else
            using AccOmp2IfAvailableElseVoid = void;
#endif
#if defined(ALPAKA_CUDA_ENABLED) && defined(__CUDACC__)
            using AccCudaIfAvailableElseVoid = AccCuda;
#else
            using AccCudaIfAvailableElseVoid = void;
#endif
            //#############################################################################
            //! A vector containing all available accelerators and void's.
            //#############################################################################
            using EnabledAccsVoid =
                boost::mpl::vector<
                    AccSerialIfAvailableElseVoid,
                    AccThreadsIfAvailableElseVoid,
                    AccFibersIfAvailableElseVoid,
                    AccOmp2IfAvailableElseVoid,
                    AccCudaIfAvailableElseVoid
                >;
        }

        //#############################################################################
        //! A vector containing all available accelerators.
        //#############################################################################
        using EnabledAccs =
            boost::mpl::filter_view<
                detail::EnabledAccsVoid,
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
            struct StreamOutAccName
            {
                template<
                    typename TAcc>
                ALPAKA_FCT_HOST_ACC auto operator()(std::ostream & os)
                -> void
                {
                    os << acc::getAccName<TAcc>();
                    os << " ";
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! Writes the enabled accelerators to the given stream.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC auto writeEnabledAccs(std::ostream & os)
        -> void
        {
            os << "Accelerators enabled: ";

            forEachType<EnabledAccs>(
                detail::StreamOutAccName(),
                std::ref(os)
                );

            os << std::endl;
        }
    }
}