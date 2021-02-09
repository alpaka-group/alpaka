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
#include <alpaka/dev/DevFpgaSyclIntel.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/TaskKernelFpgaSyclIntel.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/PltfFpgaSyclIntel.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <CL/sycl.hpp>

#include <string>

namespace alpaka
{
    template <typename TDim, typename TIdx>
    class AccFpgaSyclIntel : public AccGenericSycl<TDim, TIdx>
                           , public concepts::Implements<ConceptAcc, AccFpgaSyclIntel<TDim, TIdx>>
    {
    public:
#ifdef ALPAKA_SYCL_STREAM_ENABLED
        AccFpgaSyclIntel(Vec<TDim, TIdx> const & threadElemExtent, cl::sycl::nd_item<TDim::value> work_item,
                         cl::sycl::accessor<std::byte, 1, cl::sycl::access::mode::read_write,
                                            cl::sycl::access::target::local> shared_acc,
                         cl::sycl::stream output_stream)
        : AccGenericSycl<TDim, TIdx>(threadElemExtent, work_item, shared_acc, output_stream)
        {}
#else
        AccFpgaSyclIntel(Vec<TDim, TIdx> const & threadElemExtent, cl::sycl::nd_item<TDim::value> work_item,
                         cl::sycl::accessor<std::byte, 1, cl::sycl::access::mode::read_write,
                                            cl::sycl::access::target::local> shared_acc)
        : AccGenericSycl<TDim, TIdx>(threadElemExtent, work_item, shared_acc)
        {}
#endif

        AccFpgaSyclIntel(AccFpgaSyclIntel const&) = delete;
        auto operator=(AccFpgaSyclIntel const&) -> AccFpgaSyclIntel& = delete;

        AccFpgaSyclIntel(AccFpgaSyclIntel&&) = delete;
        auto operator=(AccFpgaSyclIntel&&) -> AccFpgaSyclIntel& = delete;

        ~AccFpgaSyclIntel() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccFpgaSyclIntel<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccFpgaSyclIntel<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The SYCL accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccFpgaSyclIntel<TDim, TIdx>>
        {
            using type = DevFpgaSyclIntel;
        };

        //#############################################################################
        //! The SYCL accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccFpgaSyclIntel<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(TWorkDiv const & workDiv, TKernelFnObj const & kernelFnObj,
                                                        TArgs const & ... args)
            {
                return TaskKernelFpgaSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>{workDiv, kernelFnObj, args...};
            }
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccFpgaSyclIntel<TDim, TIdx>>
        {
            using type = PltfFpgaSyclIntel;
        };
    }
}

#endif
