/* Copyright 2024 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/KernelBundle.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/DictTraits.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/EventPolicyList.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onAcc::internal
{
    // forward declaration to avoid cyclic includes
    template<typename T_Storage, typename T_Type>
    struct GlobalDeviceMemoryWrapper;
} // namespace alpaka::onAcc::internal

namespace alpaka::onHost
{
    namespace internal
    {
        struct MakePlatform
        {
            template<typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                auto operator()(T_Api api, T_DeviceKind deviceType) const;
            };
        };

        static auto makePlatform(auto api, alpaka::concepts::DeviceKind auto deviceType)
        {
            return MakePlatform::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceType)>{}(api, deviceType);
        }

        struct GetDeviceCount
        {
            template<typename T_Platform>
            struct Op
            {
                uint32_t operator()(T_Platform& platform) const
                {
                    return platform.getDeviceCount();
                }
            };
        };

        struct MakeDevice
        {
            template<typename T_Platform>
            struct Op
            {
                auto operator()(auto& platform, uint32_t idx) const
                {
                    return platform.makeDevice(idx);
                }
            };
        };

        struct GetDevice
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getDevice();
                }
            };
        };

        inline constexpr auto getDevice(auto&& any)
        {
            return GetDevice::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct GetNativeHandle
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getNativeHandle();
                }
            };
        };

        inline auto getNativeHandle(auto&& any)
        {
            return GetNativeHandle::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct MakeQueue
        {
            template<typename T_Device, alpaka::concepts::QueuePolicyList T_QueuePolicyList>
            struct Op
            {
                auto operator()(T_Device& device, T_QueuePolicyList const& policies) const
                {
                    return device.makeQueue(policies);
                }
            };
        };

        struct MakeEvent
        {
            template<typename T_Device, alpaka::concepts::EventPolicyList T_EventPolicyList>
            struct Op
            {
                auto operator()(T_Device& device, T_EventPolicyList const& policies) const
                {
                    return device.makeEvent(policies);
                }
            };
        };

        struct GetElapsedTime
        {
            template<typename T_StartEvent, typename T_EndEvent>
            struct Op;
        };

        struct Wait
        {
            template<typename T_Any>
            struct Op
            {
                void operator()(T_Any& any)
                {
                    any.wait();
                }
            };
        };

        inline void wait(auto&& any)
        {
            Wait::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct WaitFor
        {
            template<typename T_Queue, typename T_Event>
            struct Op
            {
                void operator()(T_Queue& queue, T_Event& event)
                {
                    queue.waitFor(event);
                }
            };
        };

        inline void waitFor(auto& queue, auto& event)
        {
            WaitFor::Op<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(event)>{}(queue, event);
        }

        struct IsEventComplete
        {
            template<typename T_Any>
            struct Op
            {
                bool operator()(T_Any& any)
                {
                    return any.isEventComplete();
                }
            };
        };

        inline bool isEventComplete(auto&& any)
        {
            return IsEventComplete::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct IsQueueEmpty
        {
            template<typename T_Queue>
            struct Op
            {
                bool operator()(T_Queue& queue)
                {
                    return queue.isQueueEmpty();
                }
            };
        };

        inline bool isQueueEmpty(auto& queue)
        {
            return IsQueueEmpty::Op<ALPAKA_TYPEOF(queue)>{}(queue);
        }

        struct Enqueue
        {
            template<
                typename T_Queue,
                onHost::concepts::ThreadOrFrameSpec T_LaunchCfg,
                alpaka::concepts::KernelBundle T_KernelBundle>
            struct Kernel
            {
                void operator()(T_Queue& queue, T_LaunchCfg const& launchCfg, T_KernelBundle const& kernelBundle) const
                {
                    queue.enqueue(launchCfg, kernelBundle);
                }
            };

            template<typename T_Queue, typename T_Task>
            struct HostTask
            {
                void operator()(T_Queue& queue, T_Task const& task) const
                {
                    queue.enqueueHostFn(task);
                }
            };

            template<typename T_Queue, typename T_Task>
            struct HostTaskDeferred
            {
                void operator()(T_Queue& queue, T_Task const& task) const
                {
                    queue.enqueueHostFnDeferred(task);
                }
            };

            template<typename T_Queue, typename T_Task>
            struct NativeFn
            {
                void operator()(T_Queue& queue, T_Task const& fn) const
                {
                    queue.enqueueNativeFn(fn);
                }
            };

            template<typename T_Queue, typename T_Event>
            struct Event
            {
                void operator()(T_Queue& queue, T_Event& event) const
                {
                    queue.enqueue(event);
                }
            };
        };

        inline void enqueueHostFn(auto& queue, auto const& task)
        {
            Enqueue::HostTask<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(task)>{}(queue, task);
        }

        inline void enqueueHostFnDeferred(auto& queue, auto const& task)
        {
            Enqueue::HostTaskDeferred<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(task)>{}(queue, task);
        }

        template<typename T_KernelFn, typename... T_Args>
        inline void enqueue(
            auto& queue,
            onHost::concepts::ThreadOrFrameSpec auto const& launchCfg,
            KernelBundle<T_KernelFn, T_Args...> const& kernelBundle)
        {
            Enqueue::Kernel<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(launchCfg), KernelBundle<T_KernelFn, T_Args...>>{}(
                queue,
                launchCfg,
                kernelBundle);
        }

        struct IsBlocking
        {
            template<::alpaka::concepts::QueueKind T>
            struct Op
            {
                constexpr bool operator()(T const&) const
                {
                    return T::isBlocking();
                }
            };
        };

        template<::alpaka::concepts::QueueKind T>
        constexpr bool isBlocking(T const& policy)
        {
            return IsBlocking::Op<ALPAKA_TYPEOF(policy)>{}(policy);
        }

        struct AdjustThreadSpec
        {
            template<
                typename T_Device,
                onHost::concepts::FrameSpec T_FrameSpec,
                alpaka::concepts::KernelBundle T_KernelBundle>
            struct Op
            {
                auto operator()(
                    T_Device const& device,
                    T_FrameSpec const& frameSpec,
                    T_KernelBundle const& kernelBundle) const
                {
                    alpaka::unused(device, frameSpec.getExecutor(), kernelBundle);
                    return ThreadSpec{frameSpec.getNumFrames(), frameSpec.getFrameExtents(), frameSpec.getExecutor()};
                }
            };
        };

        template<typename T_KernelFn, typename... T_Args>
        static auto adjustThreadSpec(
            auto const& device,
            onHost::concepts::FrameSpec auto const& frameSpec,
            KernelBundle<T_KernelFn, T_Args...> const& kernelBundle)
        {
            return AdjustThreadSpec::
                Op<ALPAKA_TYPEOF(device), ALPAKA_TYPEOF(frameSpec), KernelBundle<T_KernelFn, T_Args...>>{}(
                    device,
                    frameSpec,
                    kernelBundle);
        }

        struct Data
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return std::data(any);
                }
            };

            static decltype(auto) data(auto&& any)
            {
                return Op<ALPAKA_TYPEOF(any)>{}(any);
            }

            template<typename T_Any>
            static decltype(auto) data(Handle<T_Any>&& anyHandle)
            {
                return Op<ALPAKA_TYPEOF(*anyHandle.get())>{}(*anyHandle.get());
            }
        };

        struct Alloc
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct AllocDeferred
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct AllocUnified
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct AllocMapped
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        /** checks if a view can be accessed from the given device
         *
         * There are two paths to check if a view is accessible:
         *   - first: Try to validate the view in the scope of the device.
         *   - second: Try to validate based on soft criteria in the scope of the view's API.
         *             This path is required because the host API does not know about view data locations.
         *             The second path is optionally and will return always false if not specialized.
         */
        struct IsDataAccessible
        {
            template<typename T_Device, typename T_Any>
            struct FirstPath
            {
                bool operator()(T_Device& device, T_Any const& any) const;
            };

            template<typename T_DataApi, alpaka::concepts::DeviceKind T_DeviceKind, typename T_Any>
            struct SecondPath
            {
                bool operator()(T_DataApi, T_DeviceKind, T_Any const&) const
                {
                    return false;
                }
            };
        };

        struct Memcpy
        {
            template<typename T_Queue, typename T_Dest, typename T_Source, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, auto&&, T_Source const&, T_Extents const&) const;
            };
        };

        struct MemcpyDeviceGlobal
        {
            template<typename T_Queue, typename T_Dest, typename T_Source>
            struct Op
            {
                /** copy data from or to the device global memory
                 *
                 * It is only allowed to copy data from or to the host.
                 * Copy from device global variable to device global variables is not supported.
                 * The host data is allowed te be a host accessible pointer.
                 */
                void operator()(T_Queue& queue, T_Dest&&, T_Source&&) const;
            };
        };

        struct Memset
        {
            template<typename T_Queue, typename T_Dest, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, auto&&, uint8_t, T_Extents const&) const;
            };
        };

        struct Fill
        {
            template<typename T_Queue, typename T_Dest, typename T_Value, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, auto&&, T_Value, T_Extents const&) const;
            };
        };

        struct GetDeviceProperties
        {
            template<typename T_Any>
            struct Op
            {
                DeviceProperties operator()(auto const& platform, uint32_t idx) const;

                DeviceProperties operator()(auto const& device) const;
            };
        };

        struct GetFreeGlobalMemBytes
        {
            template<typename T_Any>
            struct Op
            {
                size_t operator()(auto const& device) const
                {
                    return device.getFreeGlobalMemBytes();
                }
            };
        };

        inline DeviceProperties getDeviceProperties(auto const& platform, uint32_t idx)
        {
            return GetDeviceProperties::Op<ALPAKA_TYPEOF(platform)>{}(platform, idx);
        }

        struct GetExtents
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.getExtents();
                }
            };
        };

        inline auto getExtents(auto&& any)
        {
            return GetExtents::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        template<typename T_Any>
        inline auto getExtents(Handle<T_Any>&& any)
        {
            return GetExtents::Op<ALPAKA_TYPEOF(*any.get())>{}(*any.get());
        }

        struct GetPitches
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.getPitches();
                }
            };
        };

        inline auto getPitches(auto&& any)
        {
            return GetPitches::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        template<typename T_Any>
        inline auto getPitches(Handle<T_Any>&& any)
        {
            return GetPitches::Op<ALPAKA_TYPEOF(*any.get())>{}(*any.get());
        }

        /** Class to get access even if the visibility is private or protected.
         *
         * In the internal implementations of alpaka most classes does not expose their members to avoid that the user
         * can easily use it in there code. Nevertheless, sometimes it is useful to ignore the visibility to simplify
         * the implementation of function interfaces. If the class you want to access defines IgnoreVisibility as
         * friend you can implement any data/function access to the class.
         *
         * @attention Do not make strong use of this method!
         */
        struct IgnoreVisibility
        {
            template<typename T_Any>
            struct Op;
        };

        /** Provide a frame specification for the given extents
         *
         * @param internalDevice must be an alpaka internal device implementation
         */
        inline constexpr auto getFrameSpec(
            auto const& internalDevice,
            alpaka::concepts::Executor auto executor,
            alpaka::concepts::VectorOrScalar auto const& extents)
        {
            static_assert(executor != exec::anyExecutor, "'exec::anyExecutor' can not be used here");
            Vec extentMd = extents;
            using ExtentVecType = ALPAKA_TYPEOF(extentMd);
            // check that all extent dimensions are greater than zero
            ALPAKA_ASSERT((extentMd > ExtentVecType::fill(0u)).reduce(std::logical_and{}));
            using IndexType = alpaka::GetValueType_t<ExtentVecType>;
            auto props = internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(internalDevice)>{}(internalDevice);
            IndexType warpSize = static_cast<IndexType>(props.warpSize);
            IndexType fastDimensionInput = std::min(warpSize, extentMd.x());
            ALPAKA_ASSERT(fastDimensionInput > IndexType{0});
            // try to create a specification with a frame size of 512 elements
            IndexType numFrameElements = 512;
            // avoid non-power of two values
            IndexType fastDimensionValue = roundDownToPowerOfTwo(fastDimensionInput);
            ExtentVecType frameExtents = ExtentVecType::fill(1).rAssign(fastDimensionValue);
            numFrameElements /= frameExtents.x();
            // distribute remainder frame elements
            while(numFrameElements > IndexType{1})
            {
                uint32_t maxIdx = ExtentVecType::dim() - 1u;
                IndexType maxValue = 0;
                for(auto i = 0u; i < ExtentVecType::dim(); ++i)
                {
                    auto v = extentMd[i] / frameExtents[i] / IndexType{2};
                    if(maxValue < v)
                    {
                        maxIdx = i;
                        maxValue = v;
                    }
                }
                // apply the change only if we not oversubscribe the extents
                auto v = extentMd[maxIdx] / frameExtents[maxIdx] / IndexType{2};
                if(v >= IndexType{1})
                    frameExtents[maxIdx] *= IndexType{2};
                else
                    break;
                numFrameElements /= IndexType{2};
            }

            ExtentVecType numFrames = divCeil(extentMd, frameExtents);
            auto frameSpec = FrameSpec{numFrames, frameExtents, executor};
            return frameSpec;
        }

        /** Provides a SIMD optimized frame specification
         *
         * The frame specification is optimized for a flat non-hierarchical execution via onAcc::worker::threadsInGrid.
         *
         * @tparam T_DataType the data type for which you would like to SIMD optimize
         * @param internalDevice must be a alpaka internal device implementation
         */
        template<typename T_DataType>
        inline constexpr auto getSimdFrameSpec(
            auto const& internalDevice,
            alpaka::concepts::Executor auto executor,
            alpaka::concepts::VectorOrScalar auto const& extents)
        {
            static_assert(executor != exec::anyExecutor, "'exec::anyExecutor' can not be used here");
            Vec extentMd = extents;
            auto deviceKind = alpaka::internal::getDeviceKind(internalDevice);
            auto deviceApi = alpaka::internal::getApi(internalDevice);
            using ExtentVecType = ALPAKA_TYPEOF(extentMd);
            // check that all extent dimensions are greater than zero
            ALPAKA_ASSERT((extentMd > ExtentVecType::fill(0u)).reduce(std::logical_and{}));
            using IndexType = alpaka::GetValueType_t<ExtentVecType>;

            ExtentVecType frameExtents = getFrameSpec(internalDevice, executor, extents).getFrameExtents();

            IndexType elementsPerFrameItem
                = static_cast<IndexType>(getNumElemPerThread<T_DataType>(deviceApi, deviceKind));

            /* The number of frames depends on an imaginary frame extent where each frame item is computing multiple
             * elements from the problem extents.
             */
            ExtentVecType numFrames
                = divExZero(extentMd, frameExtents * frameExtents.fill(1).rAssign(elementsPerFrameItem));
            // The frame specification is not required to be a multiple of the extent, it can be smaller.
            FrameSpec frameSpec = FrameSpec{numFrames, frameExtents, executor};
            return frameSpec;
        }
    } // namespace internal
} // namespace alpaka::onHost
