/* Copyright 2022 Jan Stephan, Luca Ferragina
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevCpuSyclIntel.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/kernel/TaskKernelCpuSyclIntel.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/pltf/PltfCpuSyclIntel.hpp"
#include "alpaka/pltf/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <cstddef>
#include <string>
#include <utility>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <CL/sycl.hpp>

namespace alpaka
{
    //! The Intel CPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel CPU target device.
    template<typename TDim, typename TIdx>
    class AccCpuSyclIntel final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccCpuSyclIntel<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The Intel CPU SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<AccCpuSyclIntel<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "AccCpuSyclIntel<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Intel CPU SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<AccCpuSyclIntel<TDim, TIdx>>
    {
        using type = DevCpuSyclIntel;
    };

    //! The Intel CPU SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<AccCpuSyclIntel<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return TaskKernelCpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Intel CPU SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PltfType<AccCpuSyclIntel<TDim, TIdx>>
    {
        using type = PltfCpuSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccCpuSyclIntel<TDim, TIdx>>
    {
        using type = alpaka::TagCpuSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagCpuSyclIntel, TDim, TIdx>
    {
        using type = alpaka::AccCpuSyclIntel<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
