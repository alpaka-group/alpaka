/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuUniformCudaHipRt.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/exec/UniformElements.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/wait/Traits.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <iostream>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    template<typename TApi>
    class DevUniformCudaHipRt;

    namespace detail
    {
        template<typename TElem, typename TExtent, typename TPitchBytes>
        struct FillKernelND
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                TElem* ptr,
                TElem value,
                TExtent extent,
                TPitchBytes pitchBytes) const
            {
                if(extent.prod() != 1u)
                {
                    for(auto const& idx : alpaka::uniformElementsND(acc, extent))
                    {
                        std::uintptr_t offsetBytes = static_cast<std::uintptr_t>((pitchBytes * idx).sum());

                        TElem* elem = reinterpret_cast<TElem*>(__builtin_assume_aligned(
                            reinterpret_cast<std::uint8_t*>(ptr) + offsetBytes,
                            alignof(TElem)));

                        // Write value at element address
                        *elem = value;
                    }
                }
            }
        };

        template<typename TElem, typename TExtent>
        struct FillKernel0D
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()([[maybe_unused]] TAcc const& acc, TElem* ptr, TElem value, TExtent extent)
                const
            {
                if(extent.prod() == 1u)
                {
                    TElem* elem = reinterpret_cast<TElem*>(__builtin_assume_aligned(ptr, alignof(TElem)));

                    *elem = value;
                }
            }
        };

        template<typename TDim, typename TIdx>
        TIdx getThreadNumForFill()
        {
            return 64; // tbd
        }

    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TApi>
        struct CreateTaskFill<TDim, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TViewFwd, typename TValue>
            ALPAKA_FN_HOST static auto createTaskFill(TViewFwd&& view, TValue const& value, TExtent const& extent)
            {
                using View = std::remove_reference_t<TViewFwd>;
                using Idx = alpaka::Idx<View>;
                using Acc = AccGpuUniformCudaHipRt<TApi, TDim, Idx>;
                using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
                using Vec = alpaka::Vec<TDim, Idx>;
                using Elem = alpaka::Elem<View>;
                static_assert(
                    std::is_trivially_copyable_v<Elem>,
                    "Only trivially copyable types are supported for fill");

                if constexpr(TDim::value == 0)
                {
                    Vec threads = Vec::ones();
                    Vec const elements = Vec::ones();
                    Vec blocks = Vec::ones();

                    WorkDiv grid = WorkDiv(blocks, threads, elements);
                    return alpaka::createTaskKernel<Acc>(
                        grid,
                        alpaka::detail::FillKernel0D<Elem, TExtent>{},
                        std::data(view),
                        value,
                        extent);
                }
                else
                {
                    Vec threads = Vec::ones();
                    threads.x() = alpaka::detail::getThreadNumForFill<TDim, Idx>();
                    Vec const elements = Vec::ones();
                    Vec blocks = Vec::ones();

                    WorkDiv grid = WorkDiv(blocks, threads, elements);
                    return alpaka::createTaskKernel<Acc>(
                        grid,
                        alpaka::detail::FillKernelND<Elem, TExtent, Vec>{},
                        std::data(view),
                        value,
                        extent,
                        getPitchesInBytes(view));
                }
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
