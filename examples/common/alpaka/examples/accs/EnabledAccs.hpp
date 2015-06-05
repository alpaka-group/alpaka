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

#include <alpaka/alpaka.hpp>

#include <alpaka/core/ForEachType.hpp>      // forEachType

#include <boost/mpl/vector.hpp>             // boost::mpl::vector
#include <boost/mpl/filter_view.hpp>        // boost::mpl::filter_view
#include <boost/mpl/not.hpp>                // boost::not_
#include <boost/mpl/placeholders.hpp>       // boost::mpl::_1
#include <boost/type_traits/is_same.hpp>    // boost::is_same

#include <iosfwd>                           // std::ostream

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The examples.
    //-----------------------------------------------------------------------------
    namespace examples
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
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                template<
                    typename TDim>
                using AccCpuSerialIfAvailableElseVoid = AccCpuSerial<TDim>;
#else
                template<
                    typename TDim>
                using AccCpuSerialIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
                template<
                    typename TDim>
                using AccCpuThreadsIfAvailableElseVoid = AccCpuThreads<TDim>;
#else
                template<
                    typename TDim>
                using AccCpuThreadsIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
                template<
                    typename TDim>
                using AccCpuFibersIfAvailableElseVoid = AccCpuFibers<TDim>;
#else
                template<
                    typename TDim>
                using AccCpuFibersIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                template<
                    typename TDim>
                using AccCpuOmp2BlocksIfAvailableElseVoid = AccCpuOmp2Blocks<TDim>;
#else
                template<
                    typename TDim>
                using AccCpuOmp2BlocksIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                template<
                    typename TDim>
                using AccCpuOmp2ThreadsIfAvailableElseVoid = AccCpuOmp2Threads<TDim>;
#else
                template<
                    typename TDim>
                using AccCpuOmp2ThreadsIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
                template<
                    typename TDim>
                using AccCpuOmp4CpuIfAvailableElseVoid = AccCpuOmp4<TDim>;
#else
                template<
                    typename TDim>
                using AccCpuOmp4CpuIfAvailableElseVoid = void;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
                template<
                    typename TDim>
                using AccGpuCudaIfAvailableElseVoid = AccGpuCuda<TDim>;
#else
                template<
                    typename TDim>
                using AccGpuCudaIfAvailableElseVoid = void;
#endif
                //#############################################################################
                //! A vector containing all available accelerators and void's.
                //#############################################################################
                template<
                    typename TDim>
                using EnabledAccsVoid =
                    boost::mpl::vector<
                        AccCpuSerialIfAvailableElseVoid<TDim>,
                        AccCpuThreadsIfAvailableElseVoid<TDim>,
                        AccCpuFibersIfAvailableElseVoid<TDim>,
                        AccCpuOmp2BlocksIfAvailableElseVoid<TDim>,
                        AccCpuOmp2ThreadsIfAvailableElseVoid<TDim>,
                        AccCpuOmp4CpuIfAvailableElseVoid<TDim>,
                        AccGpuCudaIfAvailableElseVoid<TDim>
                    >;
            }

            //#############################################################################
            //! A vector containing all available accelerators.
            //#############################################################################
            template<
                typename TDim>
            using EnabledAccs =
                typename boost::mpl::filter_view<
                    detail::EnabledAccsVoid<TDim>,
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
                    ALPAKA_FCT_HOST_ACC auto operator()(
                        std::ostream & os)
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
            template<
                typename TDim>
            ALPAKA_FCT_HOST_ACC auto writeEnabledAccs(
                std::ostream & os)
            -> void
            {
                os << "Accelerators enabled: ";

                forEachType<
                    EnabledAccs<TDim>>(
                        detail::StreamOutAccName(),
                        std::ref(os));

                os << std::endl;
            }
        }
    }
}
