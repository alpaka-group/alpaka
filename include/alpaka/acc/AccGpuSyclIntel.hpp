/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI)

#include <alpaka/acc/AccGenericSycl.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/DevGpuSyclIntel.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/TaskKernelGpuSyclIntel.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/PltfGpuSyclIntel.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <CL/sycl.hpp>

#include <string>

namespace alpaka
{
    template <typename TDim, typename TIdx>
    class AccGpuSyclIntel : public AccGenericSycl<TDim, TIdx>
                          , public concepts::Implements<ConceptAcc, AccGpuSyclIntel<TDim, TIdx>>
    {
    public:
#ifdef ALPAKA_SYCL_STREAM_ENABLED
        AccGpuSyclIntel(Vec<TDim, TIdx> const & threadElemExtent, cl::sycl::nd_item<TDim::value> work_item,
                        cl::sycl::accessor<std::byte, 1, cl::sycl::access::mode::read_write,
                                           cl::sycl::access::target::local> shared_acc,
                        cl::sycl::stream output_stream)
        : AccGenericSycl<TDim, TIdx>(threadElemExtent, work_item, shared_acc, output_stream)
        {}
#else
        AccGpuSyclIntel(Vec<TDim, TIdx> const & threadElemExtent, cl::sycl::nd_item<TDim::value> work_item,
                        cl::sycl::accessor<std::byte, 1, cl::sycl::access::mode::read_write,
                                           cl::sycl::access::target::local> shared_acc)
        : AccGenericSycl<TDim, TIdx>(threadElemExtent, work_item, shared_acc)
        {}
#endif

        AccGpuSyclIntel(AccGpuSyclIntel const&) = delete;        
        auto operator=(AccGpuSyclIntel const&) -> AccGpuSyclIntel& = delete;

        AccGpuSyclIntel(AccGpuSyclIntel&&) = delete;
        auto operator=(AccGpuSyclIntel&&) -> AccGpuSyclIntel& = delete;

        ~AccGpuSyclIntel() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccGpuSyclIntel<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccGpuSyclIntel<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The SYCL accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccGpuSyclIntel<TDim, TIdx>>
        {
            using type = DevGpuSyclIntel;
        };

        //#############################################################################
        //! The SYCL accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccGpuSyclIntel<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(TWorkDiv const & workDiv, TKernelFnObj const & kernelFnObj,
                                                        TArgs const & ... args)
            {
                return TaskKernelGpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>{workDiv, kernelFnObj, args...};
            }
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccGpuSyclIntel<TDim, TIdx>>
        {
            using type = PltfGpuSyclIntel;
        };
    }
}

#endif
