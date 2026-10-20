/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/concepts/api.hpp"
#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/api/cuda/computeApi.hpp"
#include "alpaka/api/generic.hpp"
#include "alpaka/api/hip/IdxLayer.hpp"
#include "alpaka/api/hip/computeApi.hpp"
#include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#include "alpaka/api/unifiedCudaHip/Event.hpp"
#include "alpaka/api/unifiedCudaHip/MemcpyKind.hpp"
#include "alpaka/api/unifiedCudaHip/concepts.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/internal/globalMem.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

#    include "alpaka/core/ApiCudaRt.hpp"

#    include <cstdint>
#    include <sstream>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        struct CallKernel;

        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
            using ApiInterface = typename T_Device::ApiInterface;

        public:
            Queue(
                internal::concepts::DeviceHandle auto device,
                uint32_t const idx,
                alpaka::concepts::QueuePolicyList auto const& policies)
                : m_device(std::move(device))
                , m_sharedCallbackThread{std::make_shared<alpaka::core::CallbackThread>()}
                , m_SharedStream{std::make_shared<Stream>(m_device)}
                , m_idx(idx)
                , m_isBlocking(internal::isBlocking(policies.getQueueKind()))
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            ~Queue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                /* Ensure that the callback thread is destroyed before the stream to flush already enqueued tasked
                 * which could submit to the stream.
                 */
                m_sharedCallbackThread.reset();
                m_SharedStream.reset();
            }

            Queue(Queue const&) = delete;
            Queue& operator=(Queue const&) = delete;

            Queue(Queue&&) = delete;
            Queue& operator=(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Queue<Queue>);
            }

            /** RAII CUDA/HIP stream
             *
             * Use this implementation via shared pointer to manage the lifetime independent of the queue and
             * callback thread.
             */
            struct Stream
            {
                Stream(Handle<T_Device> device) : m_device{std::move(device)}
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::setDevice(onHost::getNativeHandle(m_device)));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::streamCreateWithFlags(&m_stream, ApiInterface::streamNonBlocking));
                }

                ~Stream()
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                        ApiInterface,
                        ApiInterface::setDevice(onHost::getNativeHandle(m_device)));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::streamSynchronize(m_stream));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::streamDestroy(m_stream));
                }

                using Stream_t = typename ApiInterface::Stream_t;

                Stream_t getStream() const
                {
                    return m_stream;
                }

                // hold the device to avoid that the device is fully reset before the queue is destroyed
                Handle<T_Device> m_device;
                Stream_t m_stream;
            };

            Handle<T_Device> m_device;
            /* The callback thread must outlive the stream. Stream destruction synchronizes the native stream and may
             * execute pending host callbacks which submit work to this callback thread.
             */
            std::shared_ptr<core::CallbackThread> m_sharedCallbackThread;
            std::shared_ptr<Stream> m_SharedStream;
            uint32_t m_idx = 0u;
            bool m_isBlocking{false};

            /** Waits until all operations are finished depending on whether the queue is blocking or non-blocking.
             *
             * If the queue is a blocking queue the control flow will be blocked and the method is not returning until
             * all work in the queue is processed. This method should be called after the task is enqueued into the
             * native CUDA/HIP queue. There is no need to call this method before enqueuing because the queues are
             * in-order queues and even if another thread is enqueued something before the order is guaranteed.
             */
            void conditionalWait() const
            {
                if(m_isBlocking)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::streamSynchronize(getNativeHandle()));
                }
            }

            /** Waits until the event is complete if the queue is blocking.
             *
             * For a non-blocking queue this operation is a no-Op.
             */
            void conditionalWait([[maybe_unused]] typename ApiInterface::Event_t nativeEvent) const
            {
                if(m_isBlocking)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::eventSynchronize(nativeEvent));
                }
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("unifiedCudaHip::Queue id=") + std::to_string(m_idx);
            }

            friend struct onHost::internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_SharedStream->getStream();
            }

            friend struct onHost::internal::Enqueue;
            friend struct onHost::internal::Wait;

            void wait() const
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::streamSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::onHost::internal::WaitFor;

            void waitFor(unifiedCudaHip::Event<T_Device>& event)
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::streamWaitEvent(getNativeHandle(), internal::getNativeHandle(event), 0));
                // Wait for the stream only, the event should be finished after the stream operation is finished.
                conditionalWait();
            }

            friend struct internal::IsQueueEmpty;

            bool isQueueEmpty() const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);

                typename ApiInterface::Error_t ret = ApiInterface::success;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ApiInterface,
                    ret = ApiInterface::streamQuery(getNativeHandle()),
                    ApiInterface::errorNotReady);
                return (ret == ApiInterface::success);
            }

            void enqueueNativeFn(auto const& fn)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(m_device)));
                fn(getNativeHandle());
                conditionalWait();
            }

            friend struct onHost::internal::GetDevice;

            friend struct alpaka::internal::GetApi;
            friend struct onHost::internal::Memcpy;
            friend struct onHost::internal::MemcpyDeviceGlobal;
            friend struct onHost::internal::Memset;
            friend struct onHost::internal::AllocDeferred;
            friend struct CallKernel;
        };

        template<
            alpaka::concepts::Api T_Api,
            alpaka::concepts::DeviceKind T_DeviceKind,
            alpaka::concepts::Executor T_Executor,
            bool launchedWidthFrameSpec,
            typename TKernelBundle,
            typename T_OptimizedThreadSpec>
        __global__ void gpuKernel(TKernelBundle const kernelBundle, T_OptimizedThreadSpec const optimizedThreadSpec)
        {
            constexpr auto warpSizeValue = alpaka::onAcc::unifiedCudaHip::internal::WarpSize::Get<T_DeviceKind>{}();
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::unifiedCudaHip::BlockLayer{optimizedThreadSpec}),
                    DictEntry(layer::thread, onAcc::unifiedCudaHip::ThreadLayer{optimizedThreadSpec}),
                    DictEntry(object::launchedWidthFrameSpec, std::bool_constant<launchedWidthFrameSpec>{}),
                    DictEntry(action::threadBlockSync, onAcc::unifiedCudaHip::Sync{}),
                    DictEntry(object::api, T_Api{}),
                    DictEntry(object::deviceKind, T_DeviceKind{}),
                    DictEntry(object::exec, T_Executor{}),
                    DictEntry(object::warpSize, warpSizeValue)},
            };
            kernelBundle(acc);
        }

        ALPAKA_FN_HOST auto convertVecToUniformCudaHipDim(alpaka::concepts::Vector auto const& vec) -> dim3
        {
            constexpr auto vecDim = ALPAKA_TYPEOF(vec)::dim();
            dim3 dim(1, 1, 1);
            if constexpr(vecDim >= 1u)
                dim.x = static_cast<unsigned>(vec[vecDim - 1u]);
            if constexpr(vecDim >= 2u)
                dim.y = static_cast<unsigned>(vec[vecDim - 2u]);
            if constexpr(vecDim >= 3u)
                dim.z = static_cast<unsigned>(vec[vecDim - 3u]);

            return dim;
        }

        struct CallKernel
        {
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

            template<
                bool launchedWidthFrameSpec,
                typename T_Device,
                alpaka::concepts::Vector T_NumBlocks,
                alpaka::concepts::Vector T_NumThreads,
                alpaka::concepts::Executor T_Executor,
                typename T_KernelBundle>
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                ThreadSpec<T_NumBlocks, T_NumThreads, T_Executor> const& threadSpec,
                T_KernelBundle const& kernelBundle) const
            {
                static_assert(
                    ALPAKA_TYPEOF(threadSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);

                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                constexpr uint32_t dim = T_NumBlocks::dim();
                // dimension of the cuda/hip layer
                constexpr uint32_t layerDim = dim >= 4u ? 1u : dim;
                using IdxType = typename T_NumBlocks::value_type;

                Vec<IdxType, layerDim> numBlocks;
                Vec<IdxType, layerDim> numThreadsPerBlock;

                if constexpr(dim >= 4u)
                {
                    numBlocks = threadSpec.getNumBlocks().product();
                    numThreadsPerBlock = threadSpec.getNumThreads().product();
                }
                else
                {
                    numBlocks = threadSpec.getNumBlocks();
                    numThreadsPerBlock = threadSpec.getNumThreads();
                }

                using ThreadSpecType = std::conditional_t<
                    dim >= 4u,
                    ALPAKA_TYPEOF(threadSpec),
                    OptimizedThreadSpec<
                        typename ALPAKA_TYPEOF(threadSpec)::NumBlocksVecType,
                        typename ALPAKA_TYPEOF(threadSpec)::NumThreadsVecType>>;
                // thread spec which is only holding data if the dimension is larger than 3u
                auto optimizedThreadSpec = ThreadSpecType(threadSpec.getNumBlocks(), threadSpec.getNumThreads());

                auto kernelName = gpuKernel<
                    ALPAKA_TYPEOF(getApi(queue)),
                    ALPAKA_TYPEOF(getDeviceKind(queue)),
                    T_Executor,
                    launchedWidthFrameSpec,
                    T_KernelBundle,
                    ALPAKA_TYPEOF(optimizedThreadSpec)>;

                uint32_t blockDynSharedMemBytes = onHost::getDynSharedMemBytes(threadSpec, kernelBundle);

                kernelName<<<
                    convertVecToUniformCudaHipDim(numBlocks),
                    convertVecToUniformCudaHipDim(numThreadsPerBlock),
                    static_cast<std::size_t>(blockDynSharedMemBytes),
                    queue.getNativeHandle()>>>(kernelBundle, optimizedThreadSpec);

                queue.conditionalWait();
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::unifiedCudaHip::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

namespace alpaka::onHost
{
    namespace internal
    {
        template<typename T_Device, typename T_Task>
        struct Enqueue::HostTask<unifiedCudaHip::Queue<T_Device>, T_Task>
        {
            struct HostFuncData
            {
                std::shared_ptr<core::CallbackThread> sharedCallbackThread;
                T_Task t;
            };

            static void uniformCudaHipRtHostFunc(void* arg)
            {
                // take care data is deleted after the usage
                auto data = std::unique_ptr<HostFuncData>(reinterpret_cast<HostFuncData*>(arg));
                auto sharedCallbackThread = std::move(data->sharedCallbackThread);
                auto task = std::move(data->t);
                data.reset();
                auto f = sharedCallbackThread->submit([task = std::move(task)]() mutable { task(); });
                f.wait();
            }

            void operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::launchHostFunc(
                        queue.getNativeHandle(),
                        uniformCudaHipRtHostFunc,
                        new HostFuncData{queue.m_sharedCallbackThread, task}));

                queue.conditionalWait();
            }
        };

        template<typename T_Device, typename T_Task>
        struct Enqueue::HostTaskDeferred<unifiedCudaHip::Queue<T_Device>, T_Task>
        {
            // same as for Enqueue::HostTask, but not waiting for the task to finish
            struct HostFuncData
            {
                std::shared_ptr<core::CallbackThread> sharedCallbackThread;
                T_Task t;
            };

            static void uniformCudaHipRtHostFuncAsync(void* arg)
            {
                // take care data is deleted after the usage
                auto data = std::unique_ptr<HostFuncData>(reinterpret_cast<HostFuncData*>(arg));
                auto sharedCallbackThread = std::move(data->sharedCallbackThread);
                auto task = std::move(data->t);
                data.reset();
                auto f = sharedCallbackThread->submit([task = std::move(task)]() mutable { task(); });
                // don't wait, we're async
            }

            void operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::launchHostFunc(
                        queue.getNativeHandle(),
                        uniformCudaHipRtHostFuncAsync,
                        new HostFuncData{queue.m_sharedCallbackThread, task}));

                queue.conditionalWait();
            }
        };

        template<typename T_Device, typename T_Event>
        struct internal::Enqueue::Event<unifiedCudaHip::Queue<T_Device>, T_Event>
        {
            void operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Event& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                typename ApiInterface::Event_t nativeEvent = internal::getNativeHandle(event);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::eventRecord(nativeEvent, queue.getNativeHandle()));

                queue.conditionalWait(nativeEvent);
            }
        };

        template<
            typename T_Device,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            alpaka::concepts::Vector T_NumBlocks,
            alpaka::concepts::Vector T_NumThreads,
            typename T_KernelBundle>
        struct Enqueue::
            Kernel<unifiedCudaHip::Queue<T_Device>, ThreadSpec<T_NumBlocks, T_NumThreads, T_Executor>, T_KernelBundle>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                ThreadSpec<T_NumBlocks, T_NumThreads, T_Executor> const& threadSpec,
                T_KernelBundle const& kernelBundle) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                unifiedCudaHip::CallKernel{}.template operator()<false>(queue, threadSpec, kernelBundle);
            }
        };

        template<
            typename T_Device,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            typename T_NumFrames,
            typename T_FrameExtents,
            typename T_KernelBundle>
        struct Enqueue::
            Kernel<unifiedCudaHip::Queue<T_Device>, FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>, T_KernelBundle>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                FrameSpec<T_NumFrames, T_FrameExtents, T_Executor> const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                static_assert(
                    ALPAKA_TYPEOF(frameSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                auto threadBlocking = internal::adjustThreadSpec(*queue.m_device.get(), frameSpec, kernelBundle);
                unifiedCudaHip::CallKernel{}.template operator()<true>(queue, threadBlocking, kernelBundle);
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            /** Perform data copy.
             *
             * To understand the usage of pitches to shift pointers within the implementation see
             * https://alpaka.readthedocs.io/en/latest/advanced/datastorage.html#pitches
             */
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                T_Source const& src,
                T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                // use always 64bit precision to avoid overflows in the pitch calculations
                auto extentMd = pCast<size_t>(extents);

                if(extentMd.product() == size_t{0u})
                    return;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                void* destPtr = toVoidPtr(onHost::data(dest));
                auto destPitch = pCast<size_t>(onHost::getPitches(dest));
                void const* srcPtr = toVoidPtr(onHost::data(src));
                auto srcPitch = pCast<size_t>(onHost::getPitches(src));

                auto copyKind = unifiedCudaHip::MemcpyKind<
                    ApiInterface,
                    ALPAKA_TYPEOF(alpaka::internal::getApi(dest)),
                    ALPAKA_TYPEOF(alpaka::internal::getApi(src))>::kind;

                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;
                if constexpr(dim == 1u)
                {
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpyAsync(
                            destPtr,
                            srcPtr,
                            extentMd.x() * sizeof(alpaka::GetValueType_t<T_Dest>),
                            copyKind,
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpy2DAsync(
                            destPtr,
                            destPitch.y(),
                            srcPtr,
                            srcPitch.y(),
                            extentMd.x() * sizeof(alpaka::GetValueType_t<T_Dest>),
                            extentMd.y(),
                            copyKind,
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 3u)
                {
                    using VecIdxType = ALPAKA_TYPEOF(extentMd);

                    memcopy3D<alpaka::GetValueType_t<T_Dest>, ApiInterface>(
                        queue,
                        copyKind,
                        // 3D is nativ supported therefore we can handle the memcpy with a single call
                        VecIdxType::fill(1u),
                        // we do not need to adjust the src and dest pointer
                        VecIdxType::fill(0u),
                        VecIdxType::fill(0u),
                        extentMd,
                        destPtr,
                        destPitch,
                        srcPtr,
                        srcPitch);
                }
                else if constexpr(dim >= 4u)
                {
                    // Both src and dest must be contiguous memory after the 3 dimension.
                    bool isContiguous = true;
                    /* Skip the fastest two dimensions.
                     * We need to check that we do not have padding between dimension 3->4 or higher.
                     * Padding in between row and column is no problem because this is supported by CUDA/HIP.
                     */
                    for(uint32_t d = dim - 3u; d >= 1u; --d)
                    {
                        isContiguous = isContiguous && (extentMd[d] * destPitch[d] == destPitch[d - 1u])
                                       && (extentMd[d] * srcPitch[d] == srcPitch[d - 1u]);
                    }

                    if(isContiguous)
                    {
                        /* If the memory is contiguous in the dimensions higher than 3 we can emulate the N-dimensional
                         * copy with a 3D memcpy by mapping the higher dimensions into z.
                         * This is more efficient than calling the CUDA/HIP copy function multiple times.
                         */
                        alpaka::concepts::Vector<size_t, 3u> auto mappedExtentMd = extentMd.template rshrink<3u>();
                        // remove x,y dimension, fuse all other dimensions into the z component
                        mappedExtentMd.z() = extentMd.template rshrink<dim - 2u>(dim - 3u).product();
                        using VecIdxType = ALPAKA_TYPEOF(mappedExtentMd);

                        memcopy3D<alpaka::GetValueType_t<T_Dest>, ApiInterface>(
                            queue,
                            copyKind,
                            // 3D is nativ supported therefore we can handle the memcpy with a single call
                            VecIdxType::fill(1u),
                            VecIdxType::fill(0u),
                            VecIdxType::fill(0u),
                            mappedExtentMd,
                            destPtr,
                            destPitch.template rshrink<3u>(),
                            srcPtr,
                            srcPitch.template rshrink<3u>());
                    }
                    else
                    {
                        // remove the 3 fast moving dimensions
                        auto repetitions = extentMd.template rshrink<dim - 3u>(dim - 4u);
                        auto srcPitchJump = srcPitch.template rshrink<dim - 3u>(dim - 4u);
                        auto destPitchJump = destPitch.template rshrink<dim - 3u>(dim - 4u);

                        memcopy3D<alpaka::GetValueType_t<T_Dest>, ApiInterface>(
                            queue,
                            copyKind,
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

                queue.conditionalWait();
            }

            /** Memcopy which calls multiple times the 3D memcpy.
             *
             * The copy method is repetitions times repeated and the srcPtr and destPtr is advanced each time by the
             * corresponding pitches in bytes.
             *
             *  @param copyKind cuda/hip memcopy kind
             *  @param repetitions how often the 3D memcpy should be called
             *  @param destPitchJump bytes vector with pitches required to jump to the next 3D block to copy for the
             * destPtr. Dimension must be equal to repetitions.
             *  @param srcPitchJump bytes vector with pitches required to jump to the next 3D block to copy for the
             * srcPtr. Dimension must be equal to repetitions.
             *  @param extentMd Extents to describe how many elements should be copied. Dimension should be equal to
             * the original buffer/view dimension.
             *  @param destPitchesOriginal Original pitches of destPtr. Dimension should be equal to the original
             * buffer/view dimension.
             *  @param srcPitchesOriginal Original pitches of srcPtr. Dimension should be equal to the original
             * buffer/view dimension.
             */
            template<typename T_ValueType, typename T_ApiInterface>
            void memcopy3D(
                auto& queue,
                auto copyKind,
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

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    T_ApiInterface,
                    T_ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                meta::ndLoopIncIdx(
                    repetitions,
                    [&](auto const& idx)
                    {
                        // zero-init required per CUDA documentation
                        typename T_ApiInterface::Memcpy3DParms_t memCpy3DParms{};

                        memCpy3DParms.srcPtr = T_ApiInterface::makePitchedPtr(
                            // CUDA/HIP does not support const for pitched pointer
                            const_cast<uint8_t*>(
                                reinterpret_cast<uint8_t const*>(srcPtr) + (idx * srcPitchJump).sum()),
                            srcPitchesOriginal.y(),
                            extentMd.x(),
                            // number of elements in y dimension in the original memory blob
                            srcPitchesOriginal.z() / srcPitchesOriginal.y());
                        memCpy3DParms.dstPtr = T_ApiInterface::makePitchedPtr(
                            reinterpret_cast<uint8_t*>(destPtr) + (idx * destPitchJump).sum(),
                            destPitchesOriginal.y(),
                            extentMd.x(),
                            // number of elements in y dimension in the original memory blob
                            destPitchesOriginal.z() / destPitchesOriginal.y());
                        memCpy3DParms.extent = T_ApiInterface::makeExtent(
                            extentMd.x() * sizeof(T_ValueType),
                            extentMd.y(),
                            extentMd.z());
                        memCpy3DParms.kind = copyKind;

                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            T_ApiInterface,
                            T_ApiInterface::memcpy3DAsync(&memCpy3DParms, internal::getNativeHandle(queue)));
                    });
            }
        };

        // copy to device global memory
        template<typename T_Device, typename T_Source, typename T_Storage, typename T>
        struct internal::MemcpyDeviceGlobal::
            Op<unifiedCudaHip::Queue<T_Device>, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>, T_Source>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
                auto&& source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);

                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto queueApi = alpaka::internal::getApi(queue);
                auto copyKind = unifiedCudaHip::
                    MemcpyKind<ApiInterface, ALPAKA_TYPEOF(queueApi), ALPAKA_TYPEOF(api::host)>::kind;

                void* destPtr{nullptr};
                void const* srcPtr{nullptr};
                if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(source)>)
                    srcPtr = source;
                else
                    srcPtr = toVoidPtr(alpaka::onHost::data(ALPAKA_FORWARD(source)));

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::getSymbolAddress(reinterpret_cast<void**>(&destPtr), dest.getHandle(queueApi)));

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::memcpyAsync(destPtr, srcPtr, sizeof(T), copyKind, internal::getNativeHandle(queue)));

                queue.conditionalWait();
            }
        };

        // copy from device global memory
        template<typename T_Device, typename T_Dest, typename T_Storage, typename T>
        struct internal::MemcpyDeviceGlobal::
            Op<unifiedCudaHip::Queue<T_Device>, T_Dest, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);

                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                auto queueApi = alpaka::internal::getApi(queue);
                auto copyKind = unifiedCudaHip::
                    MemcpyKind<ApiInterface, ALPAKA_TYPEOF(api::host), ALPAKA_TYPEOF(queueApi)>::kind;

                void* destPtr{nullptr};
                if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(dest)>)
                    destPtr = dest;
                else
                    destPtr = toVoidPtr(alpaka::onHost::data(ALPAKA_FORWARD(dest)));

                void* srcPtr{nullptr};

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::getSymbolAddress(reinterpret_cast<void**>(&srcPtr), source.getHandle(queueApi)));

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::memcpyAsync(destPtr, srcPtr, sizeof(T), copyKind, internal::getNativeHandle(queue)));

                queue.conditionalWait();
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                uint8_t byteValue,
                T_Extents const& extents) const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                // use always 64bit precision to avoid overflows in the pitch calculations
                auto extentMd = pCast<size_t>(extents);
                if(extentMd.product() == size_t{0u})
                    return;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                void* destPtr = toVoidPtr(onHost::data(dest));
                auto destPitch = pCast<size_t>(onHost::getPitches(dest));

                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;
                if constexpr(dim == 1u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memsetAsync(
                            destPtr,
                            static_cast<int>(byteValue),
                            extentMd.x() * sizeof(alpaka::GetValueType_t<T_Dest>),
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memset2DAsync(
                            destPtr,
                            destPitch.y(),
                            static_cast<int>(byteValue),
                            extentMd.x() * sizeof(alpaka::GetValueType_t<T_Dest>),
                            extentMd.y(),
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 3u)
                {
                    using VecIdxType = ALPAKA_TYPEOF(extentMd);

                    memset3D<alpaka::GetValueType_t<T_Dest>, ApiInterface>(
                        queue,
                        byteValue,
                        // 3D is nativ supported therefore we can handle the memset with a single call
                        VecIdxType::fill(1u),
                        VecIdxType::fill(0u),
                        extentMd,
                        destPtr,
                        destPitch);
                }
                else if constexpr(dim >= 4u)
                {
                    // dest must be contiguous memory after the 3 dimension
                    bool isContiguous = true;
                    /* Skip the fastest two dimensions.
                     * We need to check that we do not have padding between dimension 3->4 or higher.
                     * Padding in between row and column is no problem because this is supported by CUDA/HIP.
                     */
                    for(uint32_t d = dim - 3u; d >= 1u; --d)
                        isContiguous = isContiguous && (extentMd[d] * destPitch[d] == destPitch[d - 1u]);

                    if(isContiguous)
                    {
                        /* If the memory is contiguous in the dimensions higher than 3 we can emulate the N-dimensional
                         * memset with a 3D memset by mapping the higher dimensions into z.
                         * This is more efficient than calling the CUDA/HIP memset function multiple times.
                         */
                        alpaka::concepts::Vector<size_t, 3u> auto mappedExtentMd = extentMd.template rshrink<3u>();
                        // remove x,y dimension, fuse all other dimensions into the z component
                        mappedExtentMd.z() = extentMd.template rshrink<dim - 2u>(dim - 3u).product();
                        using VecIdxType = ALPAKA_TYPEOF(mappedExtentMd);

                        memset3D<alpaka::GetValueType_t<T_Dest>, ApiInterface>(
                            queue,
                            byteValue,
                            // 3D is nativ supported therefore we can handle the memset with a single call
                            VecIdxType::fill(1u),
                            VecIdxType::fill(0u),
                            mappedExtentMd,
                            destPtr,
                            destPitch.template rshrink<3u>());
                    }
                    else
                    {
                        auto repetitions = extentMd.template rshrink<dim - 3u>(dim - 4u);
                        auto destPitchJump = destPitch.template rshrink<dim - 3u>(dim - 4u);

                        memset3D<alpaka::GetValueType_t<T_Dest>, ApiInterface>(
                            queue,
                            byteValue,
                            repetitions,
                            destPitchJump,
                            extentMd,
                            destPtr,
                            destPitch);
                    }
                }

                queue.conditionalWait();
            }

            /** Memset which calls multiple times the 3D memset.
             *
             * The memset method is repetitions times repeated and the destPtr is advanced each time by the
             * corresponding pitches in bytes.
             *
             *  @param repetitions how often the 3D memset should be called
             *  @param destPitchJump bytes vector with pitches required to jump to the next 3D block to memset for the
             * destPtr. Dimension must be equal to repetitions.
             *  @param extentMd Extents to describe how many elements should be set. Dimension should be equal to the
             * original buffer/view dimension.
             *  @param destPitchesOriginal Original pitches of destPtr. Dimension should be equal to the original
             * buffer/view dimension.
             */
            template<typename T_ValueType, typename T_ApiInterface>
            void memset3D(
                auto& queue,
                uint8_t byteValue,
                alpaka::concepts::Vector<size_t> auto const& repetitions,
                alpaka::concepts::Vector<size_t> auto const& destPitchJump,
                alpaka::concepts::Vector<size_t> auto const& extentMd,
                void* destPtr,
                alpaka::concepts::Vector<size_t> auto const& destPitchesOriginal) const
            {
                static_assert(ALPAKA_TYPEOF(repetitions)::dim() == ALPAKA_TYPEOF(destPitchJump)::dim());
                static_assert(ALPAKA_TYPEOF(extentMd)::dim() == ALPAKA_TYPEOF(destPitchesOriginal)::dim());

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    T_ApiInterface,
                    T_ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                meta::ndLoopIncIdx(
                    repetitions,
                    [&](auto const& idx)
                    {
                        auto const pitchedPtrVal = T_ApiInterface::makePitchedPtr(
                            reinterpret_cast<uint8_t*>(destPtr) + (idx * destPitchJump).sum(),
                            destPitchesOriginal.y(),
                            extentMd.x(),
                            destPitchesOriginal.z() / destPitchesOriginal.y());

                        auto const extentVal = T_ApiInterface::makeExtent(
                            extentMd.x() * sizeof(T_ValueType),
                            extentMd.y(),
                            extentMd.z());

                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            T_ApiInterface,
                            T_ApiInterface::memset3DAsync(
                                pitchedPtrVal,
                                static_cast<int>(byteValue),
                                extentVal,
                                internal::getNativeHandle(queue)));
                    });
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                T_Value elementValue,
                T_Extents const& extents) const
                requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                         && std::same_as<alpaka::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                // avoid that we pass a SharedBuffer and convert non alpaka data views
                auto dataView = makeView(dest);

                alpaka::internal::generic::fill(
                    queue,
                    defaultExecutor(getDevice(queue)),
                    dataView.getSubView(extents),
                    elementValue);
            }
        };

        /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
         * within a queue
         */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocDeferred::Op<T_Type, unifiedCudaHip::Queue<T_Device>, T_Extents>
        {
            auto operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                using ApiInterface = typename T_Device::ApiInterface;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                /** Each CUDA/HIP allocation is aligned to at least 128 byte but typically to 256byte
                 *
                 * @todo check if this value can be derived from the device properties
                 * @todo validate if memory is always aligned to 256 byte
                 */
                constexpr uint32_t alignment = 128u;
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                T_Type* ptr = nullptr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::mallocAsync((void**) &ptr, memSizeInByte, queue.getNativeHandle()));

                queue.conditionalWait();

                auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};

                auto deleter
                    = [ptr, sharedStream = queue.m_SharedStream, devIdx = onHost::getNativeHandle(deviceDependency)]()
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::setDevice(devIdx));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                        ApiInterface,
                        ApiInterface::freeAsync(toVoidPtr(ptr), sharedStream->getStream()));
                };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return sharedBuffer;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost
#endif
