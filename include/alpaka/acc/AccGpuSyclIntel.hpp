/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGpuSyclIntel.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/kernel/KernelBundle.hpp"
#include "alpaka/kernel/TaskKernelGpuSyclIntel.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/PlatformGpuSyclIntel.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <string>
#include <utility>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    //! The Intel GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel GPU target device.
    template<typename TDim, typename TIdx>
    class AccGpuSyclIntel final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccGpuSyclIntel<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The Intel GPU SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<AccGpuSyclIntel<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "AccGpuSyclIntel<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Intel GPU SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<AccGpuSyclIntel<TDim, TIdx>>
    {
        using type = DevGpuSyclIntel;
    };

    //! The Intel GPU SYCL accelerator execution task type trait specialization.
    //!
    //! \tparam TDim The dimensionality of the accelerator device properties.
    //! \tparam TIdx The idx type of the accelerator device properties.
    //! \tparam TWorkDiv The type of the work division.
    //! \tparam TKernelFnObj Kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<AccCpuSyclIntel<TDim, TIdx>, TWorkDiv, KernelBundle<TKernelFnObj, TArgs...>>
    {
        ALPAKA_FN_HOST static auto createTaskKernel(
            TWorkDiv const& workDiv,
            KernelBundle<TKernelFnObj, TArgs...> const& kernelBundle)
        {
            return TaskKernelCpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>(workDiv, kernelBundle);
        }
    };

    //! The Intel GPU SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PlatformType<AccGpuSyclIntel<TDim, TIdx>>
    {
        using type = PlatformGpuSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccGpuSyclIntel<TDim, TIdx>>
    {
        using type = alpaka::TagGpuSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagGpuSyclIntel, TDim, TIdx>
    {
        using type = alpaka::AccGpuSyclIntel<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
