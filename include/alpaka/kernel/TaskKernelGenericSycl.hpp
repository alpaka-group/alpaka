/* Copyright 2024 Jan Stephan, Andrea Bocci, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/KernelFunctionAttributes.hpp"
#include "alpaka/kernel/SyclSubgroupSize.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    if ALPAKA_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wunused-lambda-capture"
#        pragma clang diagnostic ignored "-Wunused-parameter"
#    endif

#    include <sycl/ext/oneapi/experimental/root_group.hpp>
#    include <sycl/ext/oneapi/properties/properties.hpp>
#    include <sycl/sycl.hpp>

#    define LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(sub_group_size)                                                    \
        if constexpr(TCooperative)                                                                                    \
        {                                                                                                             \
            cgh.parallel_for<detail::SyclKernel<TKernelFnObj, TDim, TIdx, TArgs...>>(                                 \
                sycl::nd_range<TDim::value>{global_size, local_size},                                                 \
                [item_elements, dyn_shared_accessor, st_shared_accessor, k_func, k_args](                             \
                    sycl::nd_item<TDim::value> work_item) [[sycl::reqd_sub_group_size(sub_group_size)]]               \
                {                                                                                                     \
                    auto acc = TAcc{item_elements, work_item, dyn_shared_accessor, st_shared_accessor};               \
                    std::apply(                                                                                       \
                        [k_func, &acc](typename std::decay_t<TArgs> const&... args) { k_func(acc, args...); },        \
                        k_args);                                                                                      \
                });                                                                                                   \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            cgh.parallel_for(                                                                                         \
                sycl::nd_range<TDim::value>{global_size, local_size},                                                 \
                [item_elements, dyn_shared_accessor, st_shared_accessor, k_func, k_args](                             \
                    sycl::nd_item<TDim::value> work_item) [[sycl::reqd_sub_group_size(sub_group_size)]]               \
                {                                                                                                     \
                    auto acc = TAcc{item_elements, work_item, dyn_shared_accessor, st_shared_accessor};               \
                    std::apply(                                                                                       \
                        [k_func, &acc](typename std::decay_t<TArgs> const&... args) { k_func(acc, args...); },        \
                        k_args);                                                                                      \
                });                                                                                                   \
        }

#    define LAUNCH_SYCL_KERNEL_WITH_DEFAULT_SUBGROUP_SIZE                                                             \
        if constexpr(TCooperative)                                                                                    \
        {                                                                                                             \
            cgh.parallel_for<detail::SyclKernel<TKernelFnObj, TDim, TIdx, TArgs...>>(                                 \
                sycl::nd_range<TDim::value>{global_size, local_size},                                                 \
                [item_elements, dyn_shared_accessor, st_shared_accessor, k_func, k_args](                             \
                    sycl::nd_item<TDim::value> work_item)                                                             \
                {                                                                                                     \
                    auto acc = TAcc{item_elements, work_item, dyn_shared_accessor, st_shared_accessor};               \
                    std::apply(                                                                                       \
                        [k_func, &acc](typename std::decay_t<TArgs> const&... args) { k_func(acc, args...); },        \
                        k_args);                                                                                      \
                });                                                                                                   \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            cgh.parallel_for(                                                                                         \
                sycl::nd_range<TDim::value>{global_size, local_size},                                                 \
                [item_elements, dyn_shared_accessor, st_shared_accessor, k_func, k_args](                             \
                    sycl::nd_item<TDim::value> work_item)                                                             \
                {                                                                                                     \
                    auto acc = TAcc{item_elements, work_item, dyn_shared_accessor, st_shared_accessor};               \
                    std::apply(                                                                                       \
                        [k_func, &acc](typename std::decay_t<TArgs> const&... args) { k_func(acc, args...); },        \
                        k_args);                                                                                      \
                });                                                                                                   \
        }


#    define THROW_AND_LAUNCH_EMPTY_SYCL_KERNEL                                                                        \
        throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_not_supported));                               \
        if constexpr(TCooperative)                                                                                    \
        {                                                                                                             \
            cgh.parallel_for<detail::SyclKernel<TKernelFnObj, TDim, TIdx, TArgs...>>(                                 \
                sycl::nd_range<TDim::value>{global_size, local_size},                                                 \
                [item_elements, dyn_shared_accessor, st_shared_accessor, k_func, k_args](                             \
                    sycl::nd_item<TDim::value> work_item) {});                                                        \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            cgh.parallel_for(                                                                                         \
                sycl::nd_range<TDim::value>{global_size, local_size},                                                 \
                [item_elements, dyn_shared_accessor, st_shared_accessor, k_func, k_args](                             \
                    sycl::nd_item<TDim::value> work_item) {});                                                        \
        }

namespace alpaka
{
    namespace detail
    {
        // A dummy class to pass as a template parameter when launching cooperative kernels
        template<typename TKernel, typename TDim, typename TIdx, typename... TArgs>
        class SyclKernel;
    } // namespace detail

    //! The SYCL accelerator execution task.
    template<
        concepts::Tag TTag,
        typename TAcc,
        typename TDim,
        typename TIdx,
        typename TKernelFnObj,
        bool TCooperative,
        typename... TArgs>
    class TaskKernelGenericSycl final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        static_assert(TDim::value > 0 && TDim::value <= 3, "Invalid kernel dimensionality");

        template<typename TWorkDiv>
        TaskKernelGenericSycl(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj{kernelFnObj}
            , m_args{std::forward<TArgs>(args)...}
        {
        }

        auto operator()(sycl::handler& cgh, sycl::queue const& queue) const -> void
        {
            auto const work_groups = WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
            auto const group_items = WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
            auto const item_elements = WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

            auto const global_size = get_global_size(work_groups, group_items);
            auto const local_size = get_local_size(group_items);

            // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
            auto const dyn_shared_mem_bytes = std::max(
                1ul,
                std::apply(
                    [&](std::decay_t<TArgs> const&... args) {
                        return getBlockSharedMemDynSizeBytes<TAcc>(m_kernelFnObj, group_items, item_elements, args...);
                    },
                    m_args));

            auto dyn_shared_accessor = sycl::local_accessor<std::byte>{sycl::range<1>{dyn_shared_mem_bytes}, cgh};

            // allocate static shared memory -- value comes from the build system
            constexpr auto st_shared_mem_bytes = std::size_t{ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB * 1024};
            auto st_shared_accessor = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};

            // copy-by-value so we don't access 'this' on the device
            auto k_func = m_kernelFnObj;
            auto k_args = m_args;

            constexpr std::size_t sub_group_size = trait::warpSize<TKernelFnObj, TAcc>;
            bool supported = false;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            if constexpr(TCooperative)
            {
                sycl::kernel_bundle bundle
                    = sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context());
                sycl::kernel kernel = bundle.get_kernel<class detail::SyclKernel<TKernelFnObj>>();
                size_t maxWGs = kernel.ext_oneapi_get_info<
                    sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_group_sync>(queue);
                if(work_groups.prod() > maxWGs)
                {
                    throw std::runtime_error(
                        "The number of requested blocks is larger than maximuma of the device for the kernel "
                        + core::demangled<TKernelFnObj>
                        + "! Device: " + getAccName<TAcc>() + ", requested: " + std::to_string(work_groups.prod())
                        + ", maximum allowed: " + std::to_string(maxWGs) + ". Use getMaxActiveBlocks().");
                }
#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << "maxBlocksPerGrid for the " << core::demangled<TKernelFnObj> << ": " << maxWGs
                          << std::endl;
#        endif
            }
#    endif

            if constexpr(sub_group_size == 0)
            {
                // no explicit subgroup size requirement
                LAUNCH_SYCL_KERNEL_WITH_DEFAULT_SUBGROUP_SIZE
                supported = true;
            }
            else
            {
#    if(SYCL_SUBGROUP_SIZE == 0)
                // no explicit SYCL target, assume JIT compilation
                LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(sub_group_size)
                supported = true;
#    else
                // check if the kernel should be launched with a subgroup size of 4
                if constexpr(sub_group_size == 4)
                {
#        if(SYCL_SUBGROUP_SIZE & 4)
                    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(4)
                    supported = true;
#        else
                    // empty kernel, required to keep SYCL happy
                    THROW_AND_LAUNCH_EMPTY_SYCL_KERNEL
#        endif
                }

                // check if the kernel should be launched with a subgroup size of 8
                if constexpr(sub_group_size == 8)
                {
#        if(SYCL_SUBGROUP_SIZE & 8)
                    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(8)
                    supported = true;
#        else
                    // empty kernel, required to keep SYCL happy
                    THROW_AND_LAUNCH_EMPTY_SYCL_KERNEL
#        endif
                }

                // check if the kernel should be launched with a subgroup size of 16
                if constexpr(sub_group_size == 16)
                {
#        if(SYCL_SUBGROUP_SIZE & 16)
                    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(16)
                    supported = true;
#        else
                    // empty kernel, required to keep SYCL happy
                    THROW_AND_LAUNCH_EMPTY_SYCL_KERNEL
#        endif
                }

                // check if the kernel should be launched with a subgroup size of 32
                if constexpr(sub_group_size == 32)
                {
#        if(SYCL_SUBGROUP_SIZE & 32)
                    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(32)
                    supported = true;
#        else
                    // empty kernel, required to keep SYCL happy
                    THROW_AND_LAUNCH_EMPTY_SYCL_KERNEL
#        endif
                }

                // check if the kernel should be launched with a subgroup size of 64
                if constexpr(sub_group_size == 64)
                {
#        if(SYCL_SUBGROUP_SIZE & 64)
                    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(64)
                    supported = true;
#        else
                    // empty kernel, required to keep SYCL happy
                    THROW_AND_LAUNCH_EMPTY_SYCL_KERNEL
#        endif
                }
#    endif

                // this subgroup size is not supported, raise an exception
                if(not supported)
                    throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_not_supported));
            }
        }

        static constexpr auto is_sycl_task = true;
        // Distinguish from other tasks
        static constexpr auto is_sycl_kernel = true;

    private:
        auto get_global_size(Vec<TDim, TIdx> const& work_groups, Vec<TDim, TIdx> const& group_items) const
        {
            if constexpr(TDim::value == 1)
                return sycl::range<1>{static_cast<std::size_t>(work_groups[0] * group_items[0])};
            else if constexpr(TDim::value == 2)
                return sycl::range<2>{
                    static_cast<std::size_t>(work_groups[0] * group_items[0]),
                    static_cast<std::size_t>(work_groups[1] * group_items[1])};
            else
                return sycl::range<3>{
                    static_cast<std::size_t>(work_groups[0] * group_items[0]),
                    static_cast<std::size_t>(work_groups[1] * group_items[1]),
                    static_cast<std::size_t>(work_groups[2] * group_items[2])};
        }

        auto get_local_size(Vec<TDim, TIdx> const& group_items) const
        {
            if constexpr(TDim::value == 1)
                return sycl::range<1>{static_cast<std::size_t>(group_items[0])};
            else if constexpr(TDim::value == 2)
                return sycl::range<2>{
                    static_cast<std::size_t>(group_items[0]),
                    static_cast<std::size_t>(group_items[1])};
            else
                return sycl::range<3>{
                    static_cast<std::size_t>(group_items[0]),
                    static_cast<std::size_t>(group_items[1]),
                    static_cast<std::size_t>(group_items[2])};
        }

    public:
        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

} // namespace alpaka

#    if ALPAKA_COMP_CLANG
#        pragma clang diagnostic pop
#    endif

namespace alpaka::trait
{
    //! The SYCL execution task accelerator type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct AccType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = TAcc;
    };

    //! The SYCL execution task device type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct DevType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = typename DevType<TAcc>::type;
    };

    //! The SYCL execution task platform type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct PlatformType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = typename PlatformType<TAcc>::type;
    };

    //! The SYCL execution task dimension getter trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct DimType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = TDim;
    };

    //! The SYCL execution task idx type trait specialization.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    struct IdxType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
    {
        using type = TIdx;
    };

    //! \brief Specialisation of the class template FunctionAttributes
    //! \tparam TTag The SYCL device selector.
    //! \tparam TDev The device type.
    //! \tparam TDim The dimensionality of the accelerator device properties.
    //! \tparam TIdx The idx type of the accelerator device properties.
    //! \tparam TKernelFn Kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    template<concepts::Tag TTag, typename TDev, typename TDim, typename TIdx, typename TKernelFn, typename... TArgs>
    struct FunctionAttributes<AccGenericSycl<TTag, TDim, TIdx>, TDev, TKernelFn, TArgs...>
    {
        //! \param dev The device instance
        //! \param kernelFn The kernel function object which should be executed.
        //! \param args The kernel invocation arguments.
        //! \return KernelFunctionAttributes instance. The default version always returns an instance with zero
        //! fields. For CPU, the field of max threads allowed by kernel function for the block is 1.
        ALPAKA_FN_HOST static auto getFunctionAttributes(
            TDev const& dev,
            [[maybe_unused]] TKernelFn const& kernelFn,
            [[maybe_unused]] TArgs&&... args) -> alpaka::KernelFunctionAttributes
        {
            alpaka::KernelFunctionAttributes kernelFunctionAttributes;

            // set function properties for maxThreadsPerBlock to device properties
            auto const& props = alpaka::getAccDevProps<AccGenericSycl<TTag, TDim, TIdx>>(dev);
            kernelFunctionAttributes.maxThreadsPerBlock = static_cast<int>(props.m_blockThreadCountMax);
            return kernelFunctionAttributes;
        }
    };

    //! The CUDA/HIP get max active blocks for cooperative kernel specialization.
    template<typename TAcc, typename TKernelFnObj, typename TTag, typename TDim, typename TIdx, typename... TArgs>
    struct MaxActiveBlocks<TAcc, DevGenericSycl<TTag>, TKernelFnObj, TDim, TIdx, TArgs...>
    {
        ALPAKA_FN_HOST static auto getMaxActiveBlocks(
            TKernelFnObj const& /*kernelFnObj*/,
            DevGenericSycl<TTag> const& device,
            alpaka::Vec<TDim, TIdx> const& /*blockThreadExtent*/,
            alpaka::Vec<TDim, TIdx> const& /*threadElemExtent*/,
            TArgs const&... /*args*/) -> int
        {
            sycl::queue queue{
                std::move(device.getNativeHandle()
                              .second), // This is important. In SYCL a device can belong to multiple contexts.
                std::move(device.getNativeHandle().first),
                {sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};

            sycl::kernel_bundle bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context());
            sycl::kernel kernel = bundle.get_kernel<detail::SyclKernel<TKernelFnObj, TDim, TIdx, TArgs...>>();
            size_t maxWGs = kernel.ext_oneapi_get_info<
                sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_group_sync>(queue);
            return static_cast<int>(maxWGs);
        }
    };
} // namespace alpaka::trait

#    undef LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS

#endif
