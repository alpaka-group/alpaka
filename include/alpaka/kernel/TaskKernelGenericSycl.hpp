/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Sycl.hpp>

// Avoid clang warnings that refer to this header
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#pragma clang diagnostic ignored "-Wnewline-eof"
#include <stl-tuple/STLTuple.hpp> // computecpp-sdk
#pragma clang diagnostic pop

#include <functional>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>


namespace alpaka
{
    namespace detail
    {
        template <typename TAcc, typename TKernelFnObj, typename... TArgs>
        struct kernel {}; // SYCL kernel names must be globally visible

        template <typename... TArgs, std::size_t... Is>
        constexpr auto make_device_args(std::tuple<TArgs...> args, std::index_sequence<Is...>)
        {
            return utility::tuple::make_tuple(std::get<Is>(args)...);
        }

        template <typename TKernelFnObj, typename... TArgs>
        struct kernelReturnsVoid
        {
            static constexpr auto value = std::is_same_v<std::result_of_t<TKernelFnObj(TArgs const& ...)>, void>;
        };

        template <typename TKernelFnObj, typename... TArgs>
        constexpr auto kernel_returns_void(TKernelFnObj, utility::tuple::Tuple<TArgs...> const &) -> bool
        {
            return std::is_same_v<std::result_of_t<TKernelFnObj(TArgs const & ...)>, void>;
        }

        template <typename TFunc, typename TAcc, typename... TArgs, std::size_t... Is>
        constexpr auto apply_impl(TFunc&& f, TAcc const& acc, utility::tuple::Tuple<TArgs...> t, std::index_sequence<Is...>)
        {
            f(acc, utility::tuple::get<Is>(t)...);
        }

        template <typename TFunc, typename TAcc, typename... TArgs>
        constexpr auto apply(TFunc&& f, TAcc const& acc, utility::tuple::Tuple<TArgs...> t)
        {
            apply_impl(std::forward<TFunc>(f), acc, t, std::make_index_sequence<sizeof...(TArgs)>{});
        }

        struct TaskKernelGenericSyclImpl
        {
            std::vector<cl::sycl::event> dependencies = {};
            std::shared_mutex mutex{};
        };
    } // namespace detail

    //#############################################################################
    //! The SYCL accelerator execution task.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGenericSycl final : public WorkDivMembers<TDim, TIdx>
    {
        template<typename TDev, typename TTask, typename TSfinae>
        friend struct traits::Enqueue;

    public:
        static_assert(TDim::value > 0 && TDim::value <= 3, "Invalid kernel dimensionality");
        static_assert(!std::is_same_v<TKernelFnObj, std::function<void(TAcc const&, TArgs...)>>,
                      "std::function is not allowed for SYCL kernels!");
        static_assert(alpaka::detail::kernelReturnsVoid<TKernelFnObj, TAcc, TArgs...>::value,
                      "The KernelFnObj must return void!");
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelGenericSycl(TWorkDiv && workDiv, TKernelFnObj const & kernelFnObj,
                                             TArgs const & ... args)
        : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
        , m_kernelFnObj{kernelFnObj}
        , m_args{args...}
        {
            static_assert(
                Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //-----------------------------------------------------------------------------
        TaskKernelGenericSycl(TaskKernelGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        TaskKernelGenericSycl(TaskKernelGenericSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGenericSycl const &) -> TaskKernelGenericSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGenericSycl &&) -> TaskKernelGenericSycl & = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelGenericSycl() = default;

        auto operator()(cl::sycl::handler& cgh) -> void // Don't remove the trailing void or DPCPP will complain.
        { 
            using namespace cl::sycl;

            // wait for previous kernels to complete
            cgh.depends_on(pimpl->dependencies);

            // transform arguments to device tuple so we can copy the arguments into device code. We can't use
            // std::tuple for this step because it isn't standard layout and thus prohibited to be copied into SYCL
            // device code.
            auto device_args = detail::make_device_args(m_args, std::make_index_sequence<sizeof...(TArgs)>{});

            const auto work_groups = WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
            const auto group_items = WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
            const auto item_elements = WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

            const auto global_size = get_global_size(work_groups, group_items);
            const auto local_size = get_local_size(group_items);

            // allocate shared memory -- needs at least 1 byte to make XRT happy
            const auto shared_mem_bytes = std::max(1ul, std::apply([&](auto const& ... args)
            {
                return getBlockSharedMemDynSizeBytes<TAcc>(m_kernelFnObj, group_items, item_elements, args...);
            }, m_args));

            auto shared_accessor = accessor<std::byte, 1, access::mode::read_write, access::target::local>{
                                                range<1>{shared_mem_bytes}, cgh};

            // copy-by-value so we don't access 'this' on the device
            auto k_func = m_kernelFnObj;

#ifdef ALPAKA_SYCL_STREAM_ENABLED
            // set up device-side printing
            constexpr auto buf_size = std::size_t{65535}; // 64 KiB for the output buffer
            auto buf_per_work_item = std::size_t{};
            if constexpr(TDim::value == 1)
                buf_per_work_item = buf_size / static_cast<std::size_t>(group_items[0]);
            else if constexpr(TDim::value == 2)
                buf_per_work_item = buf_size / static_cast<std::size_t>(group_items[0] * group_items[1]);
            else
                buf_per_work_item = buf_size / static_cast<std::size_t>(group_items[0] * group_items[1] * group_items[2]);

            auto output_stream = stream{buf_size, buf_per_work_item, cgh};
#endif

            cgh.parallel_for<detail::kernel<TAcc, TKernelFnObj, TArgs...>>(nd_range<TDim::value>{global_size, local_size},
            [=](nd_item<TDim::value> work_item)
            {
#ifdef ALPAKA_SYCL_STREAM_ENABLED
                auto acc = TAcc{item_elements, work_item, shared_accessor, output_stream};
#else
                auto acc = TAcc{item_elements, work_item, shared_accessor};
#endif

                alpaka::detail::apply(k_func, acc, device_args);
            });
        }

        // Distinguish from non-alpaka types (= host tasks)
        static constexpr auto is_sycl_enqueueable = true;

    private:
        auto get_global_size(const Vec<TDim, TIdx>& work_groups, const Vec<TDim, TIdx>& group_items)
        {
            using namespace cl::sycl;

            if constexpr(TDim::value == 1)
                return range<1>{static_cast<std::size_t>(work_groups[0] * group_items[0])};
            else if constexpr(TDim::value == 2)
            {
                return range<2>{static_cast<std::size_t>(work_groups[1] * group_items[1]),
                                static_cast<std::size_t>(work_groups[0] * group_items[0])};
            }
            else
            {
                return range<3>{static_cast<std::size_t>(work_groups[2] * group_items[2]),
                                static_cast<std::size_t>(work_groups[1] * group_items[1]),
                                static_cast<std::size_t>(work_groups[0] * group_items[0])};
            }
        }

        auto get_local_size(const Vec<TDim, TIdx>& group_items)
        {
            using namespace cl::sycl;

            if constexpr(TDim::value == 1)
                return range<1>{static_cast<std::size_t>(group_items[0])};
            else if constexpr(TDim::value == 2)
                return range<2>{static_cast<std::size_t>(group_items[1]), static_cast<std::size_t>(group_items[0])};
            else
            {
                return range<3>{static_cast<std::size_t>(group_items[2]), static_cast<std::size_t>(group_items[1]),
                                static_cast<std::size_t>(group_items[0])};
            }
        }

        TKernelFnObj m_kernelFnObj;
        std::tuple<TArgs...> m_args;
        std::shared_ptr<detail::TaskKernelGenericSyclImpl> pimpl{std::make_shared<detail::TaskKernelGenericSyclImpl>()};
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL execution task accelerator type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TAcc;
        };

        //#############################################################################
        //! The SYCL execution task device type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = typename DevType<TAcc>::type;
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = typename PltfType<TAcc>::type;
        };

        //#############################################################################
        //! The SYCL execution task dimension getter trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL execution task idx type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    }
}

#endif
