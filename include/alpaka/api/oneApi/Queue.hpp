/* Copyright 2025 Simeon Ehrig, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/generic.hpp"
#include "alpaka/api/oneApi/StaticSharedMemory.hpp"
#include "alpaka/api/syclGeneric/Queue.hpp"
#include "alpaka/api/syclGeneric/onAcc.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/core/syclConfig.hpp"
#include "alpaka/onAcc/internal/globalMem.hpp"
#include "alpaka/onHost/internal/interface.hpp"

#if ALPAKA_LANG_ONEAPI

#    ifndef ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS
#        define ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS 32u
#    endif

#    ifndef SYCL_EXT_ONEAPI_MEMCPY2D
#        error                                                                                                        \
            "SYCL_EXT_ONEAPI_MEMCPY2D is not defined. Extension https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_memcpy2d.asciidoc is required!"
#    endif

namespace alpaka::onHost::internal
{
    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
            const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            sycl::queue sycl_queue = queue.getNativeHandle();

            // use always 64bit precision to avoid overflows in the pitch calculations
            auto extentMd = pCast<size_t>(extents);
            if(extentMd.product() == size_t{0u})
                return;

            auto destPitch = pCast<size_t>(onHost::getPitches(dest));
            auto* destPtr = data(dest);

            constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

            sycl::event ev;

            if constexpr(dim == 2u)
            {
                ev = sycl_queue.ext_oneapi_memset2d(
                    destPtr,
                    destPitch.y(),
                    byteValue,
                    extentMd.x() * sizeof(alpaka::GetValueType_t<T_Dest>),
                    extentMd.y());
            }
            else if constexpr(dim >= 3u)
            {
                // dest must be contiguous memory after the 2 dimension
                bool isContiguous = true;

                /* Skip the fastest dimension.
                 * We need to check that we do not have padding between dimension 2->3 or higher.
                 * Padding in column or between rows is no problem because this is supported by 2d memcpy
                 */
                for(uint32_t d = dim - 2u; d >= 1u; --d)
                    isContiguous = isContiguous && (extentMd[d] * destPitch[d] == destPitch[d - 1u]);

                if(isContiguous)
                {
                    /* If the memory is contiguous in the dimensions higher than 2 we can emulate the N-dimensional
                     * memset with a 2D memset by mapping the higher dimensions into y.
                     * This is more efficient than calling the sycl memset function multiple times.
                     */
                    alpaka::concepts::Vector<size_t, 2u> auto mappedExtentMd = extentMd.template rshrink<2u>();
                    // remove x dimension, fuse all other dimensions into the y component
                    mappedExtentMd.y() = extentMd.eraseBack().product();

                    ev = memset2D<alpaka::GetValueType_t<T_Dest>>(
                        sycl_queue,
                        byteValue,
                        // 2D is nativ supported therefore we can handle the memset with a single call
                        ALPAKA_TYPEOF(mappedExtentMd)::fill(1u),
                        ALPAKA_TYPEOF(mappedExtentMd)::fill(0u),
                        mappedExtentMd,
                        destPtr,
                        destPitch.template rshrink<2u>());
                }
                else
                {
                    // remove the 2 fast moving dimensions
                    auto repetitions = extentMd.template rshrink<dim - 2u>(dim - 3u);
                    auto destPitchJump = destPitch.template rshrink<dim - 2u>(dim - 3u);

                    ev = memset2D<alpaka::GetValueType_t<T_Dest>>(
                        sycl_queue,
                        byteValue,
                        repetitions,
                        destPitchJump,
                        extentMd,
                        destPtr,
                        destPitch);
                }
            }

            queue.setLastEvent(ev);
        }

        /** Memset which calls multiple times the memset2D.
         *
         * The memset method is repetitions times repeated and the destPtr is advanced each time by the corresponding
         * pitches in bytes.
         *
         *  @param repetitions how often memset2D should be called
         *  @param destPitchJump bytes vector with pitches required to jump to the next 2D block to memset for the
         * destPtr. Dimension must be equal to repetitions.
         *  @param extentMd Extents to describe how many elements should be copied. Dimension should be equal to the
         * original buffer/view dimension.
         *  @param destPitchesOriginal Original pitches of destPtr. Dimension should be equal to the original
         * buffer/view dimension.
         */
        template<typename T_ValueType>
        sycl::event memset2D(
            sycl::queue& sycl_queue,
            uint8_t byteValue,
            alpaka::concepts::Vector<size_t> auto const& repetitions,
            alpaka::concepts::Vector<size_t> auto const& destPitchJump,
            alpaka::concepts::Vector<size_t> auto const& extentMd,
            void* destPtr,
            alpaka::concepts::Vector<size_t> auto const& destPitchesOriginal) const
        {
            static_assert(ALPAKA_TYPEOF(repetitions)::dim() == ALPAKA_TYPEOF(destPitchJump)::dim());
            static_assert(ALPAKA_TYPEOF(extentMd)::dim() == ALPAKA_TYPEOF(destPitchesOriginal)::dim());

            sycl::event ev;
            meta::ndLoopIncIdx(
                repetitions,
                [&](auto const& idx)
                {
                    if(idx != repetitions - ALPAKA_TYPEOF(repetitions)::fill(1u))
                    {
                        // Sycl is allowed to optimize and not create events for each call.
                        sycl_queue.ext_oneapi_memset2d(
                            reinterpret_cast<uint8_t*>(destPtr) + (idx * destPitchJump).sum(),
                            destPitchesOriginal.y(),
                            byteValue,
                            extentMd.x() * sizeof(T_ValueType),
                            extentMd.y());
                    }
                    else
                    {
                        // the last call must create an event
                        ev = sycl_queue.ext_oneapi_memset2d(
                            reinterpret_cast<uint8_t*>(destPtr) + (idx * destPitchJump).sum(),
                            destPitchesOriginal.y(),
                            byteValue,
                            extentMd.x() * sizeof(T_ValueType),
                            extentMd.y());
                    }
                });
            return ev;
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        /** Perform data copy.
         *
         * To understand the usage of pitches to shift pointers within the implementation see
         * https://alpaka.readthedocs.io/en/latest/advanced/datastorage.html#pitches
         */
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Source const& src,
            T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            sycl::queue sycl_queue = queue.getNativeHandle();

            // use always 64bit precision to avoid overflows in the pitch calculations
            auto extentMd = pCast<size_t>(extents);
            if(extentMd.product() == size_t{0u})
                return;

            auto* destPtr = data(dest);
            auto destPitch = pCast<size_t>(onHost::getPitches(dest));
            auto const* srcPtr = data(src);
            auto srcPitch = pCast<size_t>(onHost::getPitches(src));

            constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

            sycl::event ev;

            if constexpr(dim == 2u)
            {
                ev = sycl_queue.ext_oneapi_memcpy2d(
                    destPtr,
                    destPitch.y(),
                    srcPtr,
                    srcPitch.y(),
                    extentMd.x() * sizeof(alpaka::GetValueType_t<T_Dest>),
                    extentMd.y());
            }
            else if constexpr(dim >= 3u)
            {
                // Both src and dest must be contiguous memory after the 2 dimension
                bool isContiguous = true;

                /* Skip the fastest dimension.
                 * We need to check that we do not have padding between dimension 2->3 or higher.
                 * Padding in column or between rows is no problem because this is supported by 2d memcpy
                 */
                for(uint32_t d = dim - 2u; d >= 1u; --d)
                {
                    isContiguous = isContiguous && (extentMd[d] * destPitch[d] == destPitch[d - 1u])
                                   && (extentMd[d] * srcPitch[d] == srcPitch[d - 1u]);
                }

                if(isContiguous)
                {
                    /* If the memory is contiguous in the dimensions higher than 2 we can emulate the N-dimensional
                     * copy with a 2D memcpy by mapping the higher dimensions into y.
                     * This is more efficient than calling the sycl memcopy function multiple times.
                     */
                    alpaka::concepts::Vector<size_t, 2u> auto mappedExtentMd = extentMd.template rshrink<2u>();
                    // remove x dimension, fuse all other dimensions into the y component
                    mappedExtentMd.y() = extentMd.eraseBack().product();
                    using VecIdxType = ALPAKA_TYPEOF(mappedExtentMd);

                    ev = memcopy2D<alpaka::GetValueType_t<T_Dest>>(
                        sycl_queue,
                        // 2D is nativ supported therefore we can handle the memcpy with a single call
                        VecIdxType::fill(1u),
                        VecIdxType::fill(0u),
                        VecIdxType::fill(0u),
                        mappedExtentMd,
                        destPtr,
                        destPitch.template rshrink<2u>(),
                        srcPtr,
                        srcPitch.template rshrink<2u>());
                }
                else
                {
                    // remove the 2 fast moving dimensions
                    auto repetitions = extentMd.template rshrink<dim - 2u>(dim - 3u);
                    auto srcPitchJump = srcPitch.template rshrink<dim - 2u>(dim - 3u);
                    auto destPitchJump = destPitch.template rshrink<dim - 2u>(dim - 3u);

                    ev = memcopy2D<alpaka::GetValueType_t<T_Dest>>(
                        sycl_queue,
                        repetitions,
                        destPitchJump,
                        srcPitchJump,
                        extentMd,
                        destPtr,
                        destPitch,
                        srcPtr,
                        srcPitch);
                }
            }

            queue.setLastEvent(ev);
        }

        /** Memcopy which calls multiple times the 2D memcpy.
         *
         * The copy method is repetitions times repeated and the srcPtr and destPtr is advanced each time by the
         * corresponding pitches in bytes.
         *
         *  @param repetitions how often the 2D memcpy should be called
         *  @param destPitchJump bytes vector with pitches required to jump to the next 2D block to copy for the
         * destPtr. Dimension must be equal to repetitions.
         *  @param srcPitchJump bytes vector with pitches required to jump to the next 2D block to copy for the srcPtr.
         * Dimension must be equal to repetitions.
         *  @param extentMd Extents to describe how many elements should be copied. Dimension should be equal to the
         * original buffer/view dimension.
         *  @param destPitchesOriginal Original pitches of destPtr. Dimension should be equal to the original
         * buffer/view dimension.
         *  @param srcPitchesOriginal Original pitches of srcPtr. Dimension should be equal to the original buffer/view
         * dimension.
         */
        template<typename T_ValueType>
        sycl::event memcopy2D(
            sycl::queue& sycl_queue,
            alpaka::concepts::Vector<size_t> auto const& repetitions,
            alpaka::concepts::Vector<size_t> auto const& destPitchJump,
            alpaka::concepts::Vector<size_t> auto const& srcPitchJump,
            alpaka::concepts::Vector<size_t> auto const& extentMd,
            void* destPtr,
            alpaka::concepts::Vector<size_t> auto const& destPitchesOriginal,
            void const* srcPtr,
            alpaka::concepts::Vector<size_t> auto const& srcPitchesOriginal) const
        {
            static_assert(
                ALPAKA_TYPEOF(repetitions)::dim() == ALPAKA_TYPEOF(destPitchJump)::dim()
                && ALPAKA_TYPEOF(repetitions)::dim() == ALPAKA_TYPEOF(srcPitchJump)::dim());
            static_assert(
                ALPAKA_TYPEOF(extentMd)::dim() == ALPAKA_TYPEOF(destPitchesOriginal)::dim()
                && ALPAKA_TYPEOF(extentMd)::dim() == ALPAKA_TYPEOF(srcPitchesOriginal)::dim());

            sycl::event ev;
            meta::ndLoopIncIdx(
                repetitions,
                [&](auto const& idx)
                {
                    if(idx != repetitions - ALPAKA_TYPEOF(repetitions)::fill(1u))
                    {
                        // Sycl is allowed to optimize and not create events for each call.
                        sycl_queue.ext_oneapi_memcpy2d(
                            reinterpret_cast<uint8_t*>(destPtr) + (idx * destPitchJump).sum(),
                            destPitchesOriginal.y(),
                            reinterpret_cast<uint8_t const*>(srcPtr) + (idx * srcPitchJump).sum(),
                            srcPitchesOriginal.y(),
                            extentMd.x() * sizeof(T_ValueType),
                            extentMd.y());
                    }
                    else
                    {
                        // the last call must create an event
                        ev = sycl_queue.ext_oneapi_memcpy2d(
                            reinterpret_cast<uint8_t*>(destPtr) + (idx * destPitchJump).sum(),
                            destPitchesOriginal.y(),
                            reinterpret_cast<uint8_t const*>(srcPtr) + (idx * srcPitchJump).sum(),
                            srcPitchesOriginal.y(),
                            extentMd.x() * sizeof(T_ValueType),
                            extentMd.y());
                    }
                });

            return ev;
        }
    };

    // copy to device global memory
    template<typename T_Device, typename T_Source, typename T_Storage, typename T>
    struct internal::MemcpyDeviceGlobal::
        Op<syclGeneric::Queue<T_Device>, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>, T_Source>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
            auto&& source) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            sycl::queue sycl_queue = queue.getNativeHandle();
            void const* srcPtr{nullptr};
            if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(source)>)
                srcPtr = source;
            else
                srcPtr = toVoidPtr(alpaka::onHost::data(source));
            sycl::event ev = sycl_queue.memcpy(dest.getHandle(alpaka::api::oneApi), srcPtr);
            queue.setLastEvent(ev);
        }
    };

    // copy from device global memory
    template<typename T_Device, typename T_Dest, typename T_Storage, typename T>
    struct internal::MemcpyDeviceGlobal::
        Op<syclGeneric::Queue<T_Device>, T_Dest, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            sycl::queue sycl_queue = queue.getNativeHandle();
            void* destPtr{nullptr};
            if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(dest)>)
                destPtr = dest;
            else
                destPtr = toVoidPtr(alpaka::onHost::data(dest));
            sycl::event ev = sycl_queue.memcpy(destPtr, source.getHandle(alpaka::api::oneApi));
            queue.setLastEvent(ev);
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct internal::Fill::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Value, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Value elementValue,
            T_Extents const& extents) const
            requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                     && std::same_as<alpaka::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        {
            // avoid that we pass a SharedBuffer and convert non alpaka data views
            auto dataView = makeView(dest);

            alpaka::internal::generic::fill(
                queue,
                defaultExecutor(getDevice(queue)),
                dataView.getSubView(extents),
                elementValue);
        }
    };

    namespace detail
    {
        template<alpaka::concepts::Vector TVec>
        inline constexpr auto vecToSyclRange(TVec vec)
        {
            constexpr auto dim = std::decay_t<TVec>::dim();
            return [&vec]<auto... I>(std::index_sequence<I...>)
            // TODO: check if this is the correct order
            { return sycl::range<dim>(vec[I]...); }(std::make_index_sequence<dim>{});
        };

        template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
        struct OptimizedThreadSpec
        {
            using NumBlocksVecType = typename T_NumBlocks::UniVec;
            using NumThreadsVecType = T_NumThreads;

            static consteval uint32_t dim()
            {
                return T_NumThreads::dim();
            }

            constexpr OptimizedThreadSpec(T_NumBlocks const&, T_NumThreads const&)
            {
            }
        };

        /** provides the sycl worker description
         *
         * @return A pair of the sycl nd range and an optimized thread spec. The thread spec is not holding any data
         * for dimension smaller equal to 3u
         */
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        inline constexpr auto getWorkerDescription(T_ThreadSpec const& threadSpec)
        {
            constexpr uint32_t dim = T_ThreadSpec::dim();
            // dimension of the sycl nd range
            constexpr uint32_t syclDim = dim >= 4u ? 1u : dim;

            sycl::nd_range<syclDim> gridRange;

            if constexpr(T_ThreadSpec::dim() >= 4u)
            {
                gridRange = sycl::nd_range<syclDim>{
                    (threadSpec.getNumBlocks() * threadSpec.getNumThreads()).product(),
                    threadSpec.getNumThreads().product()};
            }
            else
            {
                gridRange = sycl::nd_range<T_ThreadSpec::dim()>{
                    detail::vecToSyclRange(threadSpec.getNumBlocks() * threadSpec.getNumThreads()),
                    detail::vecToSyclRange(threadSpec.getNumThreads())};
            }

            using ThreadSpecType = std::conditional_t<
                dim >= 4u,
                ALPAKA_TYPEOF(threadSpec),
                detail::OptimizedThreadSpec<
                    typename ALPAKA_TYPEOF(threadSpec)::NumBlocksVecType,
                    typename ALPAKA_TYPEOF(threadSpec)::NumThreadsVecType>>;
            // thread spec which is only holding data if the dimension is larger than 3u
            auto optimizedThreadSpec = ThreadSpecType(threadSpec.getNumBlocks(), threadSpec.getNumThreads());
            return std::make_pair(gridRange, optimizedThreadSpec);
        }

        /** Generate the kernel with the given warp size.
         *
         * @tparam T_dim number of dimension of the kernel
         * @tparam T_warpSize requested warp size
         * @tparam T_isValid 0u means it is not valid, else it is valid and a kernel is generated
         */
        template<uint32_t T_dim, uint32_t T_warpSize, uint32_t T_isValid>
        struct EnqueueKernelWithWarpSize
        {
            static void call(
                sycl::handler& cgh,
                auto gridRange,
                auto const& kernelBundle,
                auto const& st_shared_accessor,
                auto const& dyn_shared_accessor,
                auto const& optimizedThreadSpec,
                auto... args)
            {
                cgh.parallel_for(
                    gridRange,
                    [kernelBundle, st_shared_accessor, dyn_shared_accessor, optimizedThreadSpec, args...](
                        sycl::nd_item<T_dim> work_item) [[sycl::reqd_sub_group_size(T_warpSize)]]
                    {
                        onAcc::oneApi::StaticSharedMemory ssm(st_shared_accessor);
                        onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);

                        static_assert(T_dim > 0);
                        static_assert(T_dim <= 3, "more the 3 dimensions are not supported");
                        auto acc = onAcc::Acc{Dict{
                            DictEntry(layer::block, onAcc::syclGeneric::BlockLayer{work_item, optimizedThreadSpec}),
                            DictEntry(layer::thread, onAcc::syclGeneric::ThreadLayer{work_item, optimizedThreadSpec}),
                            DictEntry(action::threadBlockSync, onAcc::syclGeneric::Sync{work_item}),
                            DictEntry(layer::shared, std::ref(ssm)),
                            DictEntry(layer::dynShared, std::ref(dsm)),
                            DictEntry(object::dynSharedMemBytes, dsm.byte_size()),
                            args...}};

                        kernelBundle(acc);
                    });
            }
        };

        template<uint32_t T_dim, uint32_t T_warpSize>
        struct EnqueueKernelWithWarpSize<T_dim, T_warpSize, 0u>
        {
            static void call(
                [[maybe_unused]] sycl::handler& cgh,
                [[maybe_unused]] auto gridRange,
                [[maybe_unused]] auto const& kernelBundle,
                [[maybe_unused]] auto const& st_shared_accessor,
                [[maybe_unused]] auto const& dyn_shared_accessor,
                [[maybe_unused]] auto const& optimizedThreadSpec,
                [[maybe_unused]] auto... args)
            {
                printf(
                    "Dynamic evaluated warp size on host does not match the compile time warp size ( macro "
                    "ALPAKA_SYCL_SUBGROUP_SIZE) evaluated in the "
                    "kernel. Update the definition of ALPAKA_SYCL_SUBGROUP_SIZE section and check the trait "
                    "Warpsize::Dispatch<>.");
                abort();
            }
        };
    } // namespace detail

    template<
        typename T_Device,
        alpaka::concepts::Executor T_Executor,
        alpaka::concepts::Vector T_NumBlocks,
        alpaka::concepts::Vector T_NumThreads,
        alpaka::concepts::KernelBundle T_KernelBundle>
    struct Enqueue::
        Kernel<syclGeneric::Queue<T_Device>, ThreadSpec<T_NumBlocks, T_NumThreads, T_Executor>, T_KernelBundle>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            ThreadSpec<T_NumBlocks, T_NumThreads, T_Executor> const& threadSpec,
            T_KernelBundle const& kernelBundle) const
        {
            static_assert(
                ALPAKA_TYPEOF(threadSpec)::getExecutor() != exec::anyExecutor,
                "'exec::anyExecutor' can not be used to enqueue an kernel.");
            ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);

            constexpr auto st_shared_mem_bytes = onAcc::oneApi::StaticSharedMemory::sizeLookupBufferInBytes(
                ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS);
            // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
            u_int32_t blockDynSharedMemBytes
                = std::max(u_int32_t(1), onHost::getDynSharedMemBytes(threadSpec, kernelBundle));
            assert(
                st_shared_mem_bytes + blockDynSharedMemBytes
                <= queue.m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

            sycl::event ev = queue.dispatchWarpSize(
                [&](auto warpSize) requires std::same_as<
                    std::integral_constant<
                        typename ALPAKA_TYPEOF(warpSize)::value_type,
                        ALPAKA_TYPEOF(warpSize)::value>,
                    ALPAKA_TYPEOF(warpSize)>
                {
                    return queue.getNativeHandle().submit(
                        [warpSize, threadSpec, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
                        {
                            using ApiType = decltype(getApi(queue));
                            using DeviceKindType = ALPAKA_TYPEOF(getDeviceKind(queue));

                            auto st_shared_accessor
                                = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};

                            auto dyn_shared_accessor
                                = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                            auto workerDesc = detail::getWorkerDescription(threadSpec);
                            auto optimizedThreadSpec = workerDesc.second;
                            constexpr uint32_t syclDim = workerDesc.first.dimensions;

                            constexpr uint32_t w = ALPAKA_TYPEOF(warpSize)::value;
                            detail::EnqueueKernelWithWarpSize<syclDim, w, ALPAKA_SYCL_SUBGROUP_SIZE & w>::call(
                                cgh,
                                workerDesc.first,
                                kernelBundle,
                                st_shared_accessor,
                                dyn_shared_accessor,
                                optimizedThreadSpec,
                                DictEntry(object::api, ApiType{}),
                                DictEntry(object::deviceKind, DeviceKindType{}),
                                DictEntry(object::exec, T_Executor{}),
                                DictEntry(object::launchedWidthFrameSpec, std::bool_constant<false>{}),
                                DictEntry(object::warpSize, warpSize));
                        });
                });

            queue.setLastEvent(ev);
        }
    };

    template<
        typename T_Device,
        alpaka::concepts::Executor T_Executor,
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector T_FrameExtents,
        alpaka::concepts::KernelBundle T_KernelBundle>
    struct Enqueue::
        Kernel<syclGeneric::Queue<T_Device>, FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>, T_KernelBundle>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            FrameSpec<T_NumFrames, T_FrameExtents, T_Executor> const& frameSpec,
            T_KernelBundle const& kernelBundle) const
        {
            static_assert(
                ALPAKA_TYPEOF(frameSpec)::getExecutor() != exec::anyExecutor,
                "'exec::anyExecutor' can not be used to enqueue an kernel.");
            ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);

            auto const threadBlocking = internal::adjustThreadSpec(*queue.m_device.get(), frameSpec, kernelBundle);

            constexpr auto st_shared_mem_bytes = onAcc::oneApi::StaticSharedMemory::sizeLookupBufferInBytes(
                ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS);

            // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
            u_int32_t blockDynSharedMemBytes
                = std::max(u_int32_t(1), onHost::getDynSharedMemBytes(threadBlocking, kernelBundle));

            assert(
                st_shared_mem_bytes + blockDynSharedMemBytes
                <= queue.m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

            sycl::event ev = queue.dispatchWarpSize(
                [&](auto warpSize) requires std::same_as<
                    std::integral_constant<
                        typename ALPAKA_TYPEOF(warpSize)::value_type,
                        ALPAKA_TYPEOF(warpSize)::value>,
                    ALPAKA_TYPEOF(warpSize)>
                {
                    return queue.getNativeHandle().submit(
                        [warpSize, threadBlocking, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
                        {
                            using ApiType = decltype(getApi(queue));
                            using DeviceKindType = ALPAKA_TYPEOF(getDeviceKind(queue));
                            auto st_shared_accessor
                                = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};
                            auto dyn_shared_accessor
                                = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                            auto workerDesc = detail::getWorkerDescription(threadBlocking);
                            auto optimizedThreadSpec = workerDesc.second;
                            constexpr uint32_t syclDim = workerDesc.first.dimensions;

                            constexpr uint32_t w = ALPAKA_TYPEOF(warpSize)::value;

                            detail::EnqueueKernelWithWarpSize<syclDim, w, ALPAKA_SYCL_SUBGROUP_SIZE & w>::call(
                                cgh,
                                workerDesc.first,
                                kernelBundle,
                                st_shared_accessor,
                                dyn_shared_accessor,
                                optimizedThreadSpec,
                                DictEntry(object::api, ApiType{}),
                                DictEntry(object::deviceKind, DeviceKindType{}),
                                DictEntry(object::exec, T_Executor{}),
                                DictEntry(object::launchedWidthFrameSpec, std::bool_constant<true>{}),
                                DictEntry(object::warpSize, warpSize));
                        });
                });
            queue.setLastEvent(ev);
        }
    };

} // namespace alpaka::onHost::internal

#endif
