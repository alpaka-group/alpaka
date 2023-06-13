/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevFpgaSyclXilinx.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/kernel/TaskKernelFpgaSyclXilinx.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/pltf/PltfFpgaSyclXilinx.hpp"
#include "alpaka/pltf/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <string>
#include <utility>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <CL/sycl.hpp>

namespace alpaka
{
    //! The Xilinx FPGA SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a SYCL-capable Xilinx FPGA target device.
    template<typename TDim, typename TIdx>
    class AccFpgaSyclXilinx final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccFpgaSyclXilinx<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The Xilinx FPGA SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<AccFpgaSyclXilinx<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "AccFpgaSyclXilinx<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Xilinx FPGA SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<AccFpgaSyclXilinx<TDim, TIdx>>
    {
        using type = DevFpgaSyclXilinx;
    };

    //! The Xilinx FPGA SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<AccFpgaSyclXilinx<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return TaskKernelFpgaSyclXilinx<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Xilinx FPGA SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PltfType<AccFpgaSyclXilinx<TDim, TIdx>>
    {
        using type = PltfFpgaSyclXilinx;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccFpgaSyclXilinx<TDim, TIdx>>
    {
        using type = alpaka::TagFpgaSyclXilinx;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagFpgaSyclXilinx, TDim, TIdx>
    {
        using type = alpaka::AccFpgaSyclXilinx<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
