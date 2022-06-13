/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Debug.hpp>
#include <alpaka/core/DemangleTypeNames.hpp>
#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#    include <alpaka/workdiv/Traits.hpp>
#endif

#include <type_traits>

//! The alpaka accelerator library.
namespace alpaka
{
    //! The kernel traits.
    namespace trait
    {
        //! The kernel execution task creation trait.
        template<
            typename TAcc,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs/*,
            typename TSfinae = void*/>
        struct CreateTaskKernel;

        //! The trait for getting the size of the block shared dynamic memory of a kernel.
        //!
        //! \tparam TKernelFnObj The kernel function object.
        //! \tparam TAcc The accelerator.
        //!
        //! The default implementation returns 0.
        template<typename TKernelFnObj, typename TAcc, typename TSfinae = void>
        struct BlockSharedMemDynSizeBytes
        {
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
            //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
            //! \param blockThreadExtent The block thread extent.
            //! \param threadElemExtent The thread element extent.
            //! \tparam TArgs The kernel invocation argument types pack.
            //! \param args,... The kernel invocation arguments.
            //! \return The size of the shared memory allocated for a block in bytes.
            //! The default version always returns zero.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                [[maybe_unused]] TKernelFnObj const& kernelFnObj,
                [[maybe_unused]] Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                [[maybe_unused]] Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                [[maybe_unused]] TArgs const&... args) -> std::size_t
            {
                return 0u;
            }
        };

        //! The trait for getting the schedule to use when a kernel is run using the CpuOmp2Blocks accelerator.
        //!
        //! Has no effect on other accelerators.
        //!
        //! A user could either specialize this trait for their kernel, or define a public static member
        //! ompScheduleKind of type alpaka::omp::Schedule, and additionally also int member ompScheduleChunkSize. In
        //! the latter case, alpaka never odr-uses these members.
        //!
        //! In case schedule kind and chunk size are compile-time constants, setting then inside kernel may benefit
        //! performance.
        //!
        //! \tparam TKernelFnObj The kernel function object.
        //! \tparam TAcc The accelerator.
        //!
        //! The default implementation behaves as if the trait was not specialized.
        template<typename TKernelFnObj, typename TAcc, typename TSfinae = void>
        struct OmpSchedule
        {
        private:
            //! Type returned when the trait is not specialized
            struct TraitNotSpecialized
            {
            };

        public:
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
            //! \param kernelFnObj The kernel object for which the schedule should be returned.
            //! \param blockThreadExtent The block thread extent.
            //! \param threadElemExtent The thread element extent.
            //! \tparam TArgs The kernel invocation argument types pack.
            //! \param args,... The kernel invocation arguments.
            //! \return The OpenMP schedule information as an alpaka::omp::Schedule object,
            //!         returning an object of any other type is treated as if the trait is not specialized.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST static auto getOmpSchedule(
                [[maybe_unused]] TKernelFnObj const& kernelFnObj,
                [[maybe_unused]] Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                [[maybe_unused]] Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                [[maybe_unused]] TArgs const&... args) -> TraitNotSpecialized
            {
                return TraitNotSpecialized{};
            }
        };
    } // namespace trait

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //! \tparam TAcc The accelerator type.
    //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
    //! \param blockThreadExtent The block thread extent.
    //! \param threadElemExtent The thread element extent.
    //! \param args,... The kernel invocation arguments.
    //! \return The size of the shared memory allocated for a block in bytes.
    //! The default implementation always returns zero.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TKernelFnObj, typename TDim, typename... TArgs>
    ALPAKA_FN_HOST_ACC auto getBlockSharedMemDynSizeBytes(
        TKernelFnObj const& kernelFnObj,
        Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
        Vec<TDim, Idx<TAcc>> const& threadElemExtent,
        TArgs const&... args) -> std::size_t
    {
        return trait::BlockSharedMemDynSizeBytes<TKernelFnObj, TAcc>::getBlockSharedMemDynSizeBytes(
            kernelFnObj,
            blockThreadExtent,
            threadElemExtent,
            args...);
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //! \tparam TAcc The accelerator type.
    //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
    //! \param blockThreadExtent The block thread extent.
    //! \param threadElemExtent The thread element extent.
    //! \param args,... The kernel invocation arguments.
    //! \return The OpenMP schedule information as an alpaka::omp::Schedule object if the kernel specialized the
    //!         OmpSchedule trait, an object of another type if the kernel didn't specialize the trait.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TKernelFnObj, typename TDim, typename... TArgs>
    ALPAKA_FN_HOST auto getOmpSchedule(
        TKernelFnObj const& kernelFnObj,
        Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
        Vec<TDim, Idx<TAcc>> const& threadElemExtent,
        TArgs const&... args)
    {
        return trait::OmpSchedule<TKernelFnObj, TAcc>::getOmpSchedule(
            kernelFnObj,
            blockThreadExtent,
            threadElemExtent,
            args...);
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif

    namespace detail
    {
        //! Check that the return of TKernelFnObj is void
        template<typename TAcc, typename TSfinae = void>
        struct CheckFnReturnType
        {
            template<typename TKernelFnObj, typename... TArgs>
            void operator()(TKernelFnObj const&, TArgs const&...)
            {
                using Result = std::invoke_result_t<TKernelFnObj, TAcc const&, TArgs const&...>;
                static_assert(std::is_same_v<Result, void>, "The TKernelFnObj is required to return void!");
            }
        };
    } // namespace detail
    //! Creates a kernel execution task.
    //!
    //! \tparam TAcc The accelerator type.
    //! \param workDiv The index domain work division.
    //! \param kernelFnObj The kernel function object which should be executed.
    //! \param args,... The kernel invocation arguments.
    //! \return The kernel execution task.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    ALPAKA_FN_HOST auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
    {
        // check for void return type
        detail::CheckFnReturnType<TAcc>{}(kernelFnObj, args...);

#if BOOST_COMP_NVCC
        static_assert(
            std::is_trivially_copyable_v<TKernelFnObj> || __nv_is_extended_device_lambda_closure_type(TKernelFnObj)
                || __nv_is_extended_host_device_lambda_closure_type(TKernelFnObj),
            "Kernels must be trivially copyable or an extended CUDA lambda expression!");
#else
        static_assert(std::is_trivially_copyable_v<TKernelFnObj>, "Kernels must be trivially copyable!");
#endif
        static_assert(
            (std::is_trivially_copyable_v<std::decay_t<TArgs>> && ...),
            "Kernel arguments must be trivially copyable!");
        static_assert(
            Dim<std::decay_t<TWorkDiv>>::value == Dim<TAcc>::value,
            "The dimensions of TAcc and TWorkDiv have to be identical!");
        static_assert(
            std::is_same_v<Idx<std::decay_t<TWorkDiv>>, Idx<TAcc>>,
            "The idx type of TAcc and the idx type of TWorkDiv have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
        std::cout << __func__ << " workDiv: " << workDiv
                  << ", kernelFnObj: " << core::demangled<decltype(kernelFnObj)> << std::endl;
#endif
        return trait::CreateTaskKernel<TAcc, TWorkDiv, TKernelFnObj, TArgs...>::createTaskKernel(
            workDiv,
            kernelFnObj,
            std::forward<TArgs>(args)...);
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //! Executes the given kernel in the given queue.
    //!
    //! \tparam TAcc The accelerator type.
    //! \param queue The queue to enqueue the view copy task into.
    //! \param workDiv The index domain work division.
    //! \param kernelFnObj The kernel function object which should be executed.
    //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TQueue, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    ALPAKA_FN_HOST auto exec(TQueue& queue, TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        -> void
    {
        enqueue(queue, createTaskKernel<TAcc>(workDiv, kernelFnObj, std::forward<TArgs>(args)...));
    }
} // namespace alpaka
