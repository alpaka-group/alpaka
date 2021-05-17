/* Copyright 2019-2021 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/OmpSchedule.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/ApplyTuple.hpp>
#    include <alpaka/meta/Void.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <omp.h>

#    include <functional>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    include <utility>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
    namespace detail
    {
        //! Executor of parallel OpenMP loop with the given schedule
        //!
        //! Is explicitly specialized for all supported schedule kinds to help code optimization by compilers.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        //! \tparam TScheduleKind The schedule kind value.
        template<typename TKernel, typename TSchedule, omp::Schedule::Kind TScheduleKind>
        struct ParallelForImpl;

        //! Executor of parallel OpenMP loop with no schedule set
        //!
        //! Does not use chunk size.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::NoSchedule>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        /* Implementations for Static, Dynamic and Guided follow the same pattern.
         * There are two specializations of ParallelForImpl for compile-time dispatch depending on whether the
         * OmpSchedule trait is specialized.
         * The no trait case is further compile-time dispatched with a helper ParallelForStaticImpl.
         * It is based on whether ompScheduleChunkSize member is available.
         */

        //! Executor of parallel OpenMP loop with the static schedule
        //!
        //! Specialization for kernels specializing the OmpSchedule trait.
        //!
        //! \tparam TKernel The kernel type.
        template<typename TKernel>
        struct ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Static>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            //! \param schedule The schedule object.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                omp::Schedule const& schedule)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(static, schedule.chunkSize)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(static, schedule.chunkSize)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Helper executor of parallel OpenMP loop with the static schedule
        //!
        //! Generel implementation is for TKernel types without member ompScheduleChunkSize.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule, typename TSfinae = void>
        struct ParallelForStaticImpl
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(static)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(static)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Helper type to check if TKernel has member ompScheduleChunkSize
        //!
        //! Is void for those types, ill-formed otherwise.
        //!
        //! \tparam TKernel The kernel type.
        template<typename TKernel>
        using HasScheduleChunkSize = alpaka::meta::Void<decltype(TKernel::ompScheduleChunkSize)>;

        //! Helper executor of parallel OpenMP loop with the static schedule
        //!
        //! Specialization for kernels with ompScheduleChunkSize member.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForStaticImpl<TKernel, TSchedule, HasScheduleChunkSize<TKernel>>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param kernel The kernel instance reference
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const& kernel,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(static, kernel.ompScheduleChunkSize)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(static, kernel.ompScheduleChunkSize)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Executor of parallel OpenMP loop with the static schedule
        //!
        //! Specialization for kernels not specializing the OmpSchedule trait.
        //! Falls back to ParallelForStaticImpl for further dispatch.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Static> : ParallelForStaticImpl<TKernel, TSchedule>
        {
        };

        //! Executor of parallel OpenMP loop with the dynamic schedule
        //!
        //! Specialization for kernels specializing the OmpSchedule trait.
        //!
        //! \tparam TKernel The kernel type.
        template<typename TKernel>
        struct ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Dynamic>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            //! \param schedule The schedule object.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                omp::Schedule const& schedule)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(dynamic, schedule.chunkSize)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(dynamic, schedule.chunkSize)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Helper executor of parallel OpenMP loop with the dynamic schedule
        //!
        //! Generel implementation is for TKernel types without member ompScheduleChunkSize.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule, typename TSfinae = void>
        struct ParallelForDynamicImpl
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(dynamic)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(dynamic)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Helper executor of parallel OpenMP loop with the dynamic schedule
        //!
        //! Specialization for kernels with ompScheduleChunkSize member.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForDynamicImpl<TKernel, TSchedule, HasScheduleChunkSize<TKernel>>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param kernel The kernel instance reference
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const& kernel,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(dynamic, kernel.ompScheduleChunkSize)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(dynamic, kernel.ompScheduleChunkSize)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Executor of parallel OpenMP loop with the dynamic schedule
        //!
        //! Specialization for kernels not specializing the OmpSchedule trait.
        //! Falls back to ParallelForDynamicImpl for further dispatch.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Dynamic> : ParallelForDynamicImpl<TKernel, TSchedule>
        {
        };

        //! Executor of parallel OpenMP loop with the guided schedule
        //!
        //! Specialization for kernels specializing the OmpSchedule trait.
        //!
        //! \tparam TKernel The kernel type.
        template<typename TKernel>
        struct ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Guided>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            //! \param schedule The schedule object.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                omp::Schedule const& schedule)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(guided, schedule.chunkSize)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(guided, schedule.chunkSize)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Helper executor of parallel OpenMP loop with the guided schedule
        //!
        //! Generel implementation is for TKernel types without member ompScheduleChunkSize.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule, typename TSfinae = void>
        struct ParallelForGuidedImpl
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(guided)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(guided)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Helper executor of parallel OpenMP loop with the guided schedule
        //!
        //! Specialization for kernels with ompScheduleChunkSize member.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForGuidedImpl<TKernel, TSchedule, HasScheduleChunkSize<TKernel>>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param kernel The kernel instance reference
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const& kernel,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(guided, kernel.ompScheduleChunkSize)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(guided, kernel.ompScheduleChunkSize)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Executor of parallel OpenMP loop with the guided schedule
        //!
        //! Specialization for kernels not specializing the OmpSchedule trait.
        //! Falls back to ParallelForGuidedImpl for further dispatch.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Guided> : ParallelForGuidedImpl<TKernel, TSchedule>
        {
        };

#    if _OPENMP >= 200805
        //! Executor of parallel OpenMP loop with auto schedule set
        //!
        //! Does not use chunk size.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Auto>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#        pragma omp for nowait schedule(auto)
                for(TIdx i = 0; i < numIterations; ++i)
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };
#    endif

        //! Executor of parallel OpenMP loop with runtime schedule set
        //!
        //! Does not use chunk size.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Runtime>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const&,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const&)
            {
#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
                std::intmax_t i;
#        pragma omp for nowait schedule(runtime)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(runtime)
                for(TIdx i = 0; i < numIterations; ++i)
#    endif
                {
                    // Make another lambda to work around #1288
                    auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
                    wrappedLoopBody(i);
                }
            }
        };

        //! Executor of parallel OpenMP loop
        //!
        //! Performs dispatch based on schedule kind and forwards to the corresponding ParallelForImpl.
        //! The default implementation is for the kernels that do not set schedule in any way, compile-time dispatch.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule, typename TSfinae = void>
        struct ParallelFor
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param kernel The kernel instance reference
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            //! \param schedule The schedule object.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const& kernel,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const& schedule)
            {
                // Forward to ParallelForImpl that performs dispatch by by chunk size
                ParallelForImpl<TKernel, TSchedule, omp::Schedule::NoSchedule>{}(
                    kernel,
                    std::forward<TLoopBody>(loopBody),
                    numIterations,
                    schedule);
            }
        };

        //! Executor of parallel OpenMP loop
        //!
        //! Performs dispatch based on schedule kind and forwards to the corresponding ParallelForImpl.
        //! Specialization for kernels specializing the OmpSchedule trait, run-time dispatch.
        //!
        //! \tparam TKernel The kernel type.
        template<typename TKernel>
        struct ParallelFor<TKernel, omp::Schedule>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param kernel The kernel instance reference
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            //! \param schedule The schedule object.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const& kernel,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                omp::Schedule const& schedule)
            {
                // Forward to ParallelForImpl that performs dispatch by by chunk size
                switch(schedule.kind)
                {
                case omp::Schedule::NoSchedule:
                    ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::NoSchedule>{}(
                        kernel,
                        std::forward<TLoopBody>(loopBody),
                        numIterations,
                        schedule);
                    break;
                case omp::Schedule::Static:
                    ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Static>{}(
                        kernel,
                        std::forward<TLoopBody>(loopBody),
                        numIterations,
                        schedule);
                    break;
                case omp::Schedule::Dynamic:
                    ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Dynamic>{}(
                        kernel,
                        std::forward<TLoopBody>(loopBody),
                        numIterations,
                        schedule);
                    break;
                case omp::Schedule::Guided:
                    ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Guided>{}(
                        kernel,
                        std::forward<TLoopBody>(loopBody),
                        numIterations,
                        schedule);
                    break;
#    if _OPENMP >= 200805
                case omp::Schedule::Auto:
                    ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Auto>{}(
                        kernel,
                        std::forward<TLoopBody>(loopBody),
                        numIterations,
                        schedule);
                    break;
#    endif
                case omp::Schedule::Runtime:
                    ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Runtime>{}(
                        kernel,
                        std::forward<TLoopBody>(loopBody),
                        numIterations,
                        schedule);
                    break;
                }
            }
        };

        //! Helper type to check if TSchedule is a type originating from OmpSchedule trait definition
        //!
        //! \tparam TSchedule The schedule type.
        template<typename TSchedule>
        using IsOmpScheduleTraitSpecialized
            = std::integral_constant<bool, std::is_same<TSchedule, omp::Schedule>::value>;

        //! Helper type to check if member ompScheduleKind of TKernel should be used
        //!
        //! For that it has to be present, and no OmpSchedule trait specialized.
        //! Is void for those types, ill-formed otherwise.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type.
        template<typename TKernel, typename TSchedule>
        using UseScheduleKind
            = std::enable_if_t<sizeof(TKernel::ompScheduleKind) && !IsOmpScheduleTraitSpecialized<TSchedule>::value>;

        //! Executor of parallel OpenMP loop
        //!
        //! Performs dispatch based on schedule kind and forwards to the corresponding ParallelForImpl.
        //! Specialization for kernels with ompScheduleKind member, compile-time dispatch.
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        template<typename TKernel, typename TSchedule>
        struct ParallelFor<TKernel, TSchedule, UseScheduleKind<TKernel, TSchedule>>
        {
            //! Run parallel OpenMP loop
            //!
            //! \tparam TLoopBody The loop body functor type.
            //! \tparam TIdx The index type.
            //!
            //! \param kernel The kernel instance reference
            //! \param loopBody The loop body functor instance, takes iteration index as input.
            //! \param numIterations The number of loop iterations.
            //! \param schedule The schedule object.
            template<typename TLoopBody, typename TIdx>
            ALPAKA_FN_HOST void operator()(
                TKernel const& kernel,
                TLoopBody&& loopBody,
                TIdx const numIterations,
                TSchedule const& schedule)
            {
                // Forward to ParallelForImpl that performs dispatch by by chunk size
                ParallelForImpl<TKernel, TSchedule, TKernel::ompScheduleKind>{}(
                    kernel,
                    std::forward<TLoopBody>(loopBody),
                    numIterations,
                    schedule);
            }
        };

        //! Run parallel OpenMP loop
        //!
        //! \tparam TKernel The kernel type.
        //! \tparam TLoopBody The loop body functor type.
        //! \tparam TIdx The index type.
        //! \tparam TSchedule The schedule type (not necessarily omp::Schedule).
        //!
        //! \param kernel The kernel instance reference,
        //!        not perfect=forwarded to shorten SFINAE internally.
        //! \param loopBody The loop body functor instance, takes iteration index as input.
        //! \param numIterations The number of loop iterations.
        //! \param schedule The schedule object.
        template<typename TKernel, typename TLoopBody, typename TIdx, typename TSchedule>
        ALPAKA_FN_HOST ALPAKA_FN_INLINE void parallelFor(
            TKernel const& kernel,
            TLoopBody&& loopBody,
            TIdx const numIterations,
            TSchedule const& schedule)
        {
            // Forward to ParallelFor that performs first a dispatch by schedule kind, and then by chunk size
            ParallelFor<TKernel, TSchedule>{}(kernel, std::forward<TLoopBody>(loopBody), numIterations, schedule);
        }

    } // namespace detail

    //! The CPU OpenMP 2.0 block accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuOmp2Blocks final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelCpuOmp2Blocks(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }
        //-----------------------------------------------------------------------------
        TaskKernelCpuOmp2Blocks(TaskKernelCpuOmp2Blocks const&) = default;
        //-----------------------------------------------------------------------------
        TaskKernelCpuOmp2Blocks(TaskKernelCpuOmp2Blocks&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelCpuOmp2Blocks const&) -> TaskKernelCpuOmp2Blocks& = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelCpuOmp2Blocks&&) -> TaskKernelCpuOmp2Blocks& = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelCpuOmp2Blocks() = default;

        //-----------------------------------------------------------------------------
        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()() const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
            auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
            auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes = meta::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return getBlockSharedMemDynSizeBytes<AccCpuOmp2Blocks<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif
            // Bind all arguments except the accelerator.
            // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
            auto const boundKernelFnObj = meta::apply(
                [this](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return std::bind(std::ref(m_kernelFnObj), std::placeholders::_1, std::ref(args)...);
                },
                m_args);

            // The number of blocks in the grid.
            TIdx const numBlocksInGrid(gridBlockExtent.prod());
            if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
            {
                throw std::runtime_error("Only one thread per block allowed in the OpenMP 2.0 block accelerator!");
            }

            // Get the OpenMP schedule information for the given kernel and parameter types
            auto const schedule = meta::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return getOmpSchedule<AccCpuOmp2Blocks<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args);

            if(::omp_in_parallel() != 0)
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " already within a parallel region." << std::endl;
#    endif
                parallelFn(boundKernelFnObj, blockSharedMemDynSizeBytes, numBlocksInGrid, gridBlockExtent, schedule);
            }
            else
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " opening new parallel region." << std::endl;
#    endif
#    pragma omp parallel
                parallelFn(boundKernelFnObj, blockSharedMemDynSizeBytes, numBlocksInGrid, gridBlockExtent, schedule);
            }
        }

    private:
        template<typename FnObj, typename TSchedule>
        ALPAKA_FN_HOST auto parallelFn(
            FnObj const& boundKernelFnObj,
            std::size_t const& blockSharedMemDynSizeBytes,
            TIdx const& numBlocksInGrid,
            Vec<TDim, TIdx> const& gridBlockExtent,
            TSchedule const& schedule) const -> void
        {
#    pragma omp single nowait
            {
                // The OpenMP runtime does not create a parallel region when either:
                // * only one thread is required in the num_threads clause
                // * or only one thread is available
                // In all other cases we expect to be in a parallel region now.
                if((numBlocksInGrid > 1) && (::omp_get_max_threads() > 1) && (::omp_in_parallel() == 0))
                {
                    throw std::runtime_error("The OpenMP 2.0 runtime did not create a parallel region!");
                }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout << __func__ << " omp_get_num_threads: " << ::omp_get_num_threads() << std::endl;
#    endif
            }

            AccCpuOmp2Blocks<TDim, TIdx> acc(
                *static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
                blockSharedMemDynSizeBytes);

            // Body of the OpenMP parallel loop to be executed.
            // Index type is auto since we have a difference for OpenMP 2.0 and later ones
            auto loopBody = [&](auto currentIndex) {
#    if _OPENMP < 200805
                auto const i_tidx = static_cast<TIdx>(currentIndex); // for issue #840
                auto const index = Vec<DimInt<1u>, TIdx>(i_tidx); // for issue #840
#    else
                auto const index = Vec<DimInt<1u>, TIdx>(currentIndex); // for issue #840
#    endif
                acc.m_gridBlockIdx = mapIdx<TDim::value>(index, gridBlockExtent);

                boundKernelFnObj(acc);

                // After a block has been processed, the shared memory has to be deleted.
                freeSharedVars(acc);
            };

            detail::parallelFor(m_kernelFnObj, loopBody, numBlocksInGrid, schedule);
        }

        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuOmp2Blocks<TDim, TIdx>;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfCpu;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
