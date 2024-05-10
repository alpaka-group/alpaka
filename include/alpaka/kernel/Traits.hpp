/* Copyright 2023 Axel Huebl, Benjamin Worpitz, René Widera, Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber,
 *                Andrea Bocci, Aurora Perego, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Debug.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/OmpSchedule.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/KernelBundle.hpp"
#include "alpaka/kernel/KernelFunctionAttributes.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/Traits.hpp"

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
            typename TKernelBundle/*,
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

        //! \brief The structure template to access to the functions attributes of a kernel function object.
        //! \tparam TAcc The accelerator type
        //! \tparam TKernelBundle The kernel object type, which includes the kernel function object and it's invocation
        //! arguments.
        template<typename TAcc, typename TKernelBundle>
        struct FunctionAttributes
        {
            //! \param kernelBundle The kernel object instance, which includes the kernel function object and it's
            //! invocation arguments.
            //! \return KernelFunctionAttributes data structure instance. The default version always returns the
            //! instance with fields which are set to zero.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getFunctionAttributes([[maybe_unused]] TKernelBundle const& kernelBundle)
                -> alpaka::KernelFunctionAttributes
            {
                return alpaka::KernelFunctionAttributes();
            }
        };

        //! The trait for getting the warp size required by a kernel.
        //!
        //! \tparam TKernelFnObj The kernel function object.
        //! \tparam TAcc The accelerator.
        //!
        //! The default implementation returns 0, which lets the accelerator compiler and runtime choose the warp size.
        template<typename TKernelFnObj, typename TAcc, typename TSfinae = void>
        struct WarpSize : std::integral_constant<std::uint32_t, 0>
        {
        };

        //! This is a shortcut for the trait defined above
        template<typename TKernelFnObj, typename TAcc>
        inline constexpr std::uint32_t warpSize = WarpSize<TKernelFnObj, TAcc>::value;

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

    //! \tparam TKernelBundle The kernel object type, which includes the kernel function object and it's invocation
    //! arguments.
    //! \return KernelFunctionAttributes instance. Instance is filled with values returned by the accelerator API
    //! depending on the specific kernel. The default version always returns the instance with fields which are set to
    //! zero.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TKernelBundle>
    ALPAKA_FN_HOST_ACC auto getFunctionAttributes(TKernelBundle const& kernelBundle)
        -> alpaka::KernelFunctionAttributes
    {
        return trait::FunctionAttributes<TAcc, TKernelBundle>::getFunctionAttributes(kernelBundle);
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

    //! Trait used for overload type ambiguity of exec function
    //!
    template<typename T>
    struct is_kernel_bundle
    {
        static bool const value = false; // Compile time Constant
    };

    //! Trait used for overload type ambiguity of exec function.
    //!
    template<typename TKernelFn, typename... TArgs>
    struct is_kernel_bundle<KernelBundle<TKernelFn, TArgs...>>
    {
        static bool const value = true;
    };


#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
//! Creates a kernel execution task.
//!
//! \tparam TAcc The accelerator type.
//! \param workDiv The index domain work division.
//! \param kernelBundle.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TWorkDiv, typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST auto createTaskKernel(
        TWorkDiv const& workDiv,
        KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
    {
        static_assert(
            Dim<std::decay_t<TWorkDiv>>::value == Dim<TAcc>::value,
            "The dimensions of TAcc and TWorkDiv have to be identical!");
        static_assert(
            std::is_same_v<Idx<std::decay_t<TWorkDiv>>, Idx<TAcc>>,
            "The idx type of TAcc and the idx type of TWorkDiv have to be identical!");

        // Return the task kernel
        return trait::CreateTaskKernel<TAcc, TWorkDiv, KernelBundle<TKernelFn, TArgs...>>::createTaskKernel(
            workDiv,
            kernelBundle);
    }


#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
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
    template<typename TAcc, typename TWorkDiv, typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST auto createTaskKernel(TWorkDiv const& workDiv, TKernelFn const& kernelFnObj, TArgs const&... args)
    {
        static_assert(
            Dim<std::decay_t<TWorkDiv>>::value == Dim<TAcc>::value,
            "The dimensions of TAcc and TWorkDiv have to be identical!");
        static_assert(
            std::is_same_v<Idx<std::decay_t<TWorkDiv>>, Idx<TAcc>>,
            "The idx type of TAcc and the idx type of TWorkDiv have to be identical!");
        KernelBundle<TKernelFn, TArgs...> const& kernelBundle = alpaka::KernelBundle(kernelFnObj, args...);
        // Return the task kernel
        return trait::CreateTaskKernel<TAcc, TWorkDiv, KernelBundle<TKernelFn, TArgs...>>::createTaskKernel(
            workDiv,
            kernelBundle);
    }

    //! Executes the given kernel in the given queue.
    //!
    //! \tparam TAcc The accelerator type.
    //! \tparam TQueue The queue type for work to be submitted to.
    //! \tparam TWorkDiv The type of the work division.
    //! \tparam TKernelFn Kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    //! \param queue The queue to enqueue the view copy task into.
    //! \param workDiv The index domain work division.
    //! \param kernelBundle The kernel object instance, which includes the kernel function object and it's
    //! invocation arguments.
    //!
    template<typename TAcc, typename TQueue, typename TWorkDiv, typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST auto exec(
        TQueue& queue,
        TWorkDiv const& workDiv,
        KernelBundle<TKernelFn, TArgs...> const& kernelBundle) -> void
    {
        enqueue(queue, createTaskKernel<TAcc>(workDiv, kernelBundle));
    }

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //! Executes the given kernel in the given queue.
    //!
//! \tparam TAcc The accelerator type.
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TWorkDiv The type of the work division.
//! \tparam TKernelFn Kernel function object type.
//! \tparam TArgs Kernel function object argument types as a parameter pack.
//! \param queue The queue to enqueue the view copy task into.
//! \param workDiv The index domain work division.
//! \param kernelFnObj The kernel function object which should be executed.
//! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TAcc, typename TQueue, typename TWorkDiv, typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST auto exec(TQueue& queue, TWorkDiv const& workDiv, TKernelFn const& kernelFnObj, TArgs... args)
        -> void
    {
        auto const& bundeledKernel = alpaka::KernelBundle(kernelFnObj, args...);
        enqueue(queue, createTaskKernel<TAcc>(workDiv, bundeledKernel));
    }

} // namespace alpaka
