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

#include <alpaka/core/ForEachType.hpp>      // core::forEachType

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
                    typename TDim,
                    typename TSize>
                using AccCpuSerialIfAvailableElseVoid = acc::AccCpuSerial<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuSerialIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuThreadsIfAvailableElseVoid = acc::AccCpuThreads<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuThreadsIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuFibersIfAvailableElseVoid = acc::AccCpuFibers<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuFibersIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2BlocksIfAvailableElseVoid = acc::AccCpuOmp2Blocks<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2BlocksIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2ThreadsIfAvailableElseVoid = acc::AccCpuOmp2Threads<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2ThreadsIfAvailableElseVoid = void;
#endif
#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp4IfAvailableElseVoid = acc::AccCpuOmp4<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp4IfAvailableElseVoid = void;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
                template<
                    typename TDim,
                    typename TSize>
                using AccGpuCudaRtIfAvailableElseVoid = acc::AccGpuCudaRt<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccGpuCudaRtIfAvailableElseVoid = void;
#endif
                //#############################################################################
                //! A vector containing all available accelerators and void's.
                //#############################################################################
                template<
                    typename TDim,
                    typename TSize>
                using EnabledAccsVoid =
                    boost::mpl::vector<
                        AccCpuSerialIfAvailableElseVoid<TDim, TSize>,
                        AccCpuThreadsIfAvailableElseVoid<TDim, TSize>,
                        AccCpuFibersIfAvailableElseVoid<TDim, TSize>,
                        AccCpuOmp2BlocksIfAvailableElseVoid<TDim, TSize>,
                        AccCpuOmp2ThreadsIfAvailableElseVoid<TDim, TSize>,
                        AccCpuOmp4IfAvailableElseVoid<TDim, TSize>,
                        AccGpuCudaRtIfAvailableElseVoid<TDim, TSize>
                    >;
            }

            //#############################################################################
            //! A vector containing all available accelerators.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            using EnabledAccs =
                typename boost::mpl::filter_view<
                    detail::EnabledAccsVoid<TDim, TSize>,
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
                    ALPAKA_NO_HOST_ACC_WARNING
                    template<
                        typename TAcc>
                    ALPAKA_FN_HOST_ACC auto operator()(
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
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TDim,
                typename TSize>
            ALPAKA_FN_HOST_ACC auto writeEnabledAccs(
                std::ostream & os)
            -> void
            {
                os << "Accelerators enabled: ";

                core::forEachType<
                    EnabledAccs<TDim, TSize>>(
                        detail::StreamOutAccName(),
                        std::ref(os));

                os << std::endl;
            }
        }

        namespace detail
        {
            //#############################################################################
            //! The stream type trait for the stream that should be used for the given accelerator.
            //#############################################################################
            template<
                typename TDev,
                typename TSfinae = void>
            struct StreamType
            {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::stream::StreamCpuSync;
#else
                using type = alpaka::stream::StreamCpuAsync;
#endif
            };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
            //#############################################################################
            //! The stream type trait specialization for the CUDA accelerator.
            //#############################################################################
            template<>
            struct StreamType<
                alpaka::dev::DevCudaRt>
            {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                using type = alpaka::stream::StreamCudaRtSync;
#else
                using type = alpaka::stream::StreamCudaRtAsync;
#endif
            };
#endif
        }
        //#############################################################################
        //! The stream type that should be used for the given accelerator.
        //#############################################################################
        template<
            typename TAcc>
        using Stream = typename detail::StreamType<TAcc>::type;
    }
}
