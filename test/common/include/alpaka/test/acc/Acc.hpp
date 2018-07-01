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

#include <tuple>
#include <type_traits>
#include <iosfwd>

// When compiling the tests with CUDA enabled (nvcc or native clang) on the CI infrastructure
// we have to dramatically reduce the number of tested combinations.
// Else the log length would be exceeded.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA && defined(ALPAKA_CI)
    #define ALPAKA_CUDA_CI
#endif

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test accelerator specifics.
        namespace acc
        {
            //-----------------------------------------------------------------------------
            //! The detail namespace is used to separate implementation details from user accessible code.
            namespace detail
            {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuSerialIfAvailableElseInt = alpaka::acc::AccCpuSerial<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuSerialIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuThreadsIfAvailableElseInt = alpaka::acc::AccCpuThreads<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuFibersIfAvailableElseInt = alpaka::acc::AccCpuFibers<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuFibersIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuTbbIfAvailableElseInt = alpaka::acc::AccCpuTbbBlocks<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuTbbIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2BlocksIfAvailableElseInt = alpaka::acc::AccCpuOmp2Blocks<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2BlocksIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2ThreadsIfAvailableElseInt = alpaka::acc::AccCpuOmp2Threads<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2ThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_BT_OMP4_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp4IfAvailableElseInt = alpaka::acc::AccCpuOmp4<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp4IfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_BT_OPENACC2_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOpenAcc2IfAvailableElseInt = alpaka::acc::AccCpuOpenAcc2<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOpenAcc2IfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuCudaRtIfAvailableElseInt = alpaka::acc::AccGpuCudaRt<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuCudaRtIfAvailableElseInt = int;
#endif
                //#############################################################################
                //! A vector containing all available accelerators and void's.
                template<
                    typename TDim,
                    typename TIdx>
                using EnabledAccsElseInt =
                    std::tuple<
                        AccCpuSerialIfAvailableElseInt<TDim, TIdx>,
                        AccCpuThreadsIfAvailableElseInt<TDim, TIdx>,
                        AccCpuFibersIfAvailableElseInt<TDim, TIdx>,
                        AccCpuTbbIfAvailableElseInt<TDim, TIdx>,
                        AccCpuOmp2BlocksIfAvailableElseInt<TDim, TIdx>,
                        AccCpuOmp2ThreadsIfAvailableElseInt<TDim, TIdx>,
                        AccCpuOmp4IfAvailableElseInt<TDim, TIdx>,
                        AccCpuOpenAcc2IfAvailableElseInt<TDim, TIdx>,
                        AccGpuCudaRtIfAvailableElseInt<TDim, TIdx>
                    >;
            }

            //#############################################################################
            //! A vector containing all available accelerators.
            template<
                typename TDim,
                typename TIdx>
            using EnabledAccs =
                typename alpaka::meta::Filter<
                    detail::EnabledAccsElseInt<TDim, TIdx>,
                    std::is_class
                >;

            namespace detail
            {
                //#############################################################################
                //! The accelerator name write wrapper.
                struct StreamOutAccName
                {
                    template<
                        typename TAcc>
                    ALPAKA_FN_HOST auto operator()(
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
            template<
                typename TDim,
                typename TIdx>
            ALPAKA_FN_HOST auto writeEnabledAccs(
                std::ostream & os)
            -> void
            {
                os << "Accelerators enabled: ";

                alpaka::meta::forEachType<
                    EnabledAccs<TDim, TIdx>>(
                        detail::StreamOutAccName(),
                        std::ref(os));

                os << std::endl;
            }

#if defined(ALPAKA_CUDA_CI)
            //#############################################################################
            //! A std::tuple holding dimensions.
            using TestDims =
                std::tuple<
                    alpaka::dim::DimInt<1u>,
                    //alpaka::dim::DimInt<2u>,
                    alpaka::dim::DimInt<3u>
            // The CUDA acceleator does not currently support 4D buffers and 4D acceleration.
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA)
                    /*,alpaka::dim::DimInt<4u>*/
#endif
                >;

            //#############################################################################
            //! A std::tuple holding idx types.
            using TestIdxs =
                std::tuple<
                    std::size_t,
                    //std::int64_t,
                    std::uint64_t,
                    std::int32_t,
                    //std::uint32_t,
                    //std::int16_t,
                    std::uint16_t/*,
                    // When Idx is a 8 bit integer, extents within the tests would be extremely limited
                    // (especially when Dim is 4). Therefore, we do not test it.
                    std::int8_t,
                    std::uint8_t*/>;
#else

            //#############################################################################
            //! A std::tuple holding dimensions.
            using TestDims =
                std::tuple<
                    alpaka::dim::DimInt<1u>,
                    alpaka::dim::DimInt<2u>,
                    alpaka::dim::DimInt<3u>
            // The CUDA acceleator does not currently support 4D buffers and 4D acceleration.
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA)
                    ,alpaka::dim::DimInt<4u>
#endif
                >;

            //#############################################################################
            //! A std::tuple holding idx types.
            using TestIdxs =
                std::tuple<
                    // size_t is most probably identical to either std::uint64_t or std::uint32_t.
                    // This would lead to duplicate tests (especially test names) which is not allowed.
                    //std::size_t,
                    std::int64_t,
                    std::uint64_t,
                    std::int32_t,
                    std::uint32_t,
                    //std::int16_t,
                    std::uint16_t/*,
                    // When Idx is a 8 bit integer, extents within the tests would be extremely limited
                    // (especially when Dim is 4). Therefore, we do not test it.
                    std::int8_t,
                    std::uint8_t*/>;
#endif

            namespace detail
            {
                //#############################################################################
                //! A std::tuple holding multiple std::tuple consisting of a dimension and a idx type.
                //!
                //! TestDimIdxTuples =
                //!     tuple<
                //!         tuple<Dim1,Idx1>,
                //!         tuple<Dim2,Idx1>,
                //!         tuple<Dim3,Idx1>,
                //!         ...,
                //!         tuple<DimN,IdxN>>
                using TestDimIdxTuples =
                    alpaka::meta::CartesianProduct<
                        std::tuple,
                        TestDims,
                        TestIdxs
                    >;

                //#############################################################################
                //! Transforms a std::tuple holding a dimension and a idx type to a fully instantiated accelerator.
                //!
                //! EnabledAccs<Dim,Idx> = tuple<Acc1<Dim,Idx>, ..., AccN<Dim,Idx>>
                template<
                    typename TTestAccParamSet>
                struct InstantiateEnabledAccsWithTestParamSetImpl
                {
                    using type =
                        EnabledAccs<
                            typename std::tuple_element<0, TTestAccParamSet>::type,
                            typename std::tuple_element<1, TTestAccParamSet>::type
                        >;
                };

                template<
                    typename TTestAccParamSet>
                using InstantiateEnabledAccsWithTestParamSet = typename InstantiateEnabledAccsWithTestParamSetImpl<TTestAccParamSet>::type;

                //#############################################################################
                //! A std::tuple containing std::tuple with fully instantiated accelerators.
                //!
                //! TestEnabledAccs =
                //!     tuple<
                //!         tuple<Acc1<Dim1,Idx1>, ..., AccN<Dim1,Idx1>>,
                //!         tuple<Acc1<Dim2,Idx1>, ..., AccN<Dim2,Idx1>>,
                //!         ...,
                //!         tuple<Acc1<DimN,IdxN>, ..., AccN<DimN,IdxN>>>
                using InstantiatedEnabledAccs =
                    alpaka::meta::Transform<
                        TestDimIdxTuples,
                        InstantiateEnabledAccsWithTestParamSet
                    >;
            }

            //#############################################################################
            //! A std::tuple containing fully instantiated accelerators.
            //!
            //! TestAccs =
            //!     tuple<
            //!         Acc1<Dim1,Idx1>, ..., AccN<Dim1,Idx1>,
            //!         Acc1<Dim2,Idx1>, ..., AccN<Dim2,Idx1>,
            //!         ...,
            //!         Acc1<DimN,IdxN>, ..., AccN<DimN,IdxN>>
            using TestAccs =
                alpaka::meta::Apply<
                    detail::InstantiatedEnabledAccs,
                    alpaka::meta::Concatenate
                >;
        }
    }
}
