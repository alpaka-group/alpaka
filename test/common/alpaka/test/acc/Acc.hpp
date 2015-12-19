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

#include <alpaka/meta/ForEachType.hpp>      // meta::forEachType

#include <tuple>                            // std::tuple
#include <type_traits>                      // std::is_class
#include <iosfwd>                           // std::ostream

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    //-----------------------------------------------------------------------------
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test accelerator specifics.
        //-----------------------------------------------------------------------------
        namespace acc
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
                using AccCpuSerialIfAvailableElseVoid = alpaka::acc::AccCpuSerial<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuSerialIfAvailableElseVoid = int;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuThreadsIfAvailableElseVoid = alpaka::acc::AccCpuThreads<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuThreadsIfAvailableElseVoid = int;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuFibersIfAvailableElseVoid = alpaka::acc::AccCpuFibers<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuFibersIfAvailableElseVoid = int;
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2BlocksIfAvailableElseVoid = alpaka::acc::AccCpuOmp2Blocks<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2BlocksIfAvailableElseVoid = int;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2ThreadsIfAvailableElseVoid = alpaka::acc::AccCpuOmp2Threads<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp2ThreadsIfAvailableElseVoid = int;
#endif
#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp4IfAvailableElseVoid = alpaka::acc::AccCpuOmp4<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccCpuOmp4IfAvailableElseVoid = int;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
                template<
                    typename TDim,
                    typename TSize>
                using AccGpuCudaRtIfAvailableElseVoid = alpaka::acc::AccGpuCudaRt<TDim, TSize>;
#else
                template<
                    typename TDim,
                    typename TSize>
                using AccGpuCudaRtIfAvailableElseVoid = int;
#endif
                //#############################################################################
                //! A vector containing all available accelerators and void's.
                //#############################################################################
                template<
                    typename TDim,
                    typename TSize>
                using EnabledAccsVoid =
                    std::tuple<
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
                typename alpaka::meta::Filter<
                    detail::EnabledAccsVoid<TDim, TSize>,
                    std::is_class
                >;

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
                        os << alpaka::acc::getAccName<TAcc>();
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

                meta::forEachType<
                    EnabledAccs<TDim, TSize>>(
                        detail::StreamOutAccName(),
                        std::ref(os));

                os << std::endl;
            }
        }
    }
}
