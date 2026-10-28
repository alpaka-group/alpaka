/* Copyright 2025 René Widera
 * SPDX-License-Identifier: Apache-2.0
 */


#include "alpaka/alpaka.hpp"

using namespace alpaka;

// BEGIN-CHEATSHEET-myKernel
struct MyKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const&, [[maybe_unused]] auto... kernelArgs) const
    {
    }
};

// END-CHEATSHEET-myKernel

struct Kernel
{
    Kernel(int initElement) : element(initElement)
    {
    }

    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, [[maybe_unused]] auto... kernelArgs) const
    {
        using DataType = float;

        // BEGIN-CHEATSHEET-staticSharedMem
        // two-dimensional matrix with 4 columns, 3 rows with elements of the type float
        concepts::IMdSpan auto sharedMdArray
            = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, CVec<uint32_t, 3, 4>{});
        // or with a preprocessor unique id
        concepts::IMdSpan auto sharedMdArray2
            = alpaka::onAcc::declareSharedMdArray<float, __COUNTER__>(acc, CVec<uint32_t, 3, 4>{});
        // a single scalar
        DataType scalar = alpaka::onAcc::declareSharedVar<float, alpaka::uniqueId()>(acc, CVec<uint32_t, 3, 4>{});
        // END-CHEATSHEET-staticSharedMem

        unused(scalar);
    }

    int element;
};

struct FooDynMemKernel
{
    using DataType = int;

    // BEGIN-CHEATSHEET-dynSharedMem
    struct DynMemKernel
    {
        uint32_t dynSharedMemBytes = 32u;

        ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
        {
            // Access within the kernel, it is a plain pointer.
            // You are responsible to guarantee in bounds accesses.
            [[maybe_unused]] DataType* dynS = onAcc::getDynSharedMem<DataType>(acc);
        }
    };

    // END-CHEATSHEET-dynSharedMem
};

// BEGIN-CHEATSHEET-dynSharedMemTrait
struct DynSharedMemTrait
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
    {
        // Access within the kernel, it is a plain pointer.
        // You are responsible to guarantee in bounds accesses.
        [[maybe_unused]] int* dynS = onAcc::getDynSharedMem<int>(acc);
    }
};

// specialization within the host code
namespace alpaka::onHost::trait
{
    template<concepts::ThreadSpec T_ThreadSpec>
    struct BlockDynSharedMemBytes<DynSharedMemTrait, T_ThreadSpec>
    {
        BlockDynSharedMemBytes(DynSharedMemTrait const& kernel, T_ThreadSpec const& spec)
        {
            alpaka::unused(kernel, spec);
        }

        // the signature is very similar to the kernel operator() signature with the difference that no accelerator is
        // provided.
        uint32_t operator()([[maybe_unused]] auto const&... args) const
        {
            return 32;
        }
    };
} // namespace alpaka::onHost::trait

// END-CHEATSHEET-dynSharedMemTrait


struct KernelWait
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
    {
        // BEGIN-CHEATSHEET-inKernelBlockWait
        onAcc::syncBlockThreads(acc);
        // END-CHEATSHEET-inKernelBlockWait

        int* ptr = nullptr;
        // BEGIN-CHEATSHEET-atomicAdd
        // Operation: operation::Add, operation::Sub, operation::Min, operation::Max, operation::Exch,
        //            operation::Inc, operation::Dec, operation::And, operation::Or, operation::Xor,
        //            operation::Cas
        using Operation = operation::Add;
        auto result = atomicOp<Operation>(acc, ptr, 1);
        // Also dedicated functions available, e.g.:
        auto old = onAcc::atomicAdd(acc, ptr, 1);
        // END-CHEATSHEET-atomicAdd

        float argument = 1.0;
        float base = 1.0;
        float exp = 1.0;
        // BEGIN-CHEATSHEET-math
        [[maybe_unused]] auto sinValue = math::sin(argument);
        [[maybe_unused]] auto cosValue = math::pow(base, exp);
        // END-CHEATSHEET-math
    }
};

// Minimal example demonstrating memory fences at different scopes
struct MemoryFenceKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
    {
        // BEGIN-CHEATSHEET-memFence
        // Scopes: All threads of the block, the device and the system(host and peer devices)
        onAcc::memFence(acc, onAcc::scope::block, onAcc::order::acquire);
        onAcc::memFence(acc, onAcc::scope::device, onAcc::order::release);
        onAcc::memFence(acc, onAcc::scope::system, onAcc::order::acqRel);
        // END-CHEATSHEET-memFence
    }
};

auto main() -> int
{
    // BEGIN-CHEATSHEET-init
    // 1u, 2u, 3u, ...
    constexpr uint32_t dim = 2u;
    // uint32_t, size_t
    using IdxType = size_t;
    using DataType = int;
    // END-CHEATSHEET-init

    unused(dim, IdxType{}, DataType{});

    constexpr uint32_t numElements = 100;
    constexpr uint32_t value = 42;
    constexpr uint32_t valueX = 42;
    constexpr uint32_t valueY = 43;
    constexpr uint32_t valueZ = 44;

    unused(numElements, value);

    // BEGIN-CHEATSHEET-vectorCreate
    // Use alpaka vector as a static array for the extents
    concepts::Vector auto extent1D = Vec{value};
    concepts::Vector auto extent2D = Vec{valueY, valueX};
    // truly compile time known values
    concepts::CVector auto extent3D = CVec<IdxType, valueZ, valueY, valueX>{};
    // END-CHEATSHEET-vectorCreate

    unused(extent1D, extent2D);

    {
        // BEGIN-CHEATSHEET-vectorAccess
        auto extentZ = extent3D[0];
        auto extentX = extent3D[2];
        auto [z, y, x] = extent3D;
        // END-CHEATSHEET-vectorAccess

        unused(extentZ, extentX, x, y, z);
    }

    {
        concepts::CVector auto idx3D = CVec<IdxType, valueZ, valueY, valueX>{};

        // BEGIN-CHEATSHEET-linearize
        std::integral auto linearIdx = linearize(extent3D, idx3D);
        // END-CHEATSHEET-linearize

        unused(linearIdx);

        int scalar = 2;
        concepts::CVector auto extent4D = CVec<int, valueZ, valueZ, valueY, valueX>{};
        // BEGIN-CHEATSHEET-mapToMd
        concepts::Vector auto idxMd = mapToND(extent4D, scalar);
        // END-CHEATSHEET-mapToMd

        unused(idxMd);
    }

    uint32_t index = 0;

    auto task = []() {};
    try
    {
        // BEGIN-CHEATSHEET-makeDevice
        concepts::Api auto const api = api::host;
        concepts::DeviceKind auto const deviceKind = deviceKind::cpu;
        auto devSelector = api + deviceKind;
        if(devSelector.getDeviceCount() == 0)
            throw std::runtime_error("No device found!");
        auto device = devSelector.makeDevice(index);
        // END-CHEATSHEET-makeDevice

        // BEGIN-CHEATSHEET-makeQueue
        // default queue is non blocking
        auto queue = device.makeQueue();
        auto nonBlockingQueue = device.makeQueue(queueKind::nonBlocking);
        auto blockingQueue = device.makeQueue(queueKind::blocking);
        // END-CHEATSHEET-makeQueue

        // BEGIN-CHEATSHEET-enqueueHostTask
        queue.enqueueHostFn(task);
        queue.enqueueHostFnDeferred(task);
        // END-CHEATSHEET-enqueueHostTask

        // BEGIN-CHEATSHEET-isQueueEmpty
        bool isQueueEmpty = queue.isEmpty();
        // END-CHEATSHEET-isQueueEmpty
        alpaka::unused(isQueueEmpty);

        // BEGIN-CHEATSHEET-waitQueue
        onHost::wait(queue);
        // END-CHEATSHEET-waitQueue

        // BEGIN-CHEATSHEET-makeEvent
        auto event = device.makeEvent();
        // END-CHEATSHEET-makeEvent

        // BEGIN-CHEATSHEET-enqueueEvent
        queue.enqueue(event);
        // END-CHEATSHEET-enqueueEvent

        // BEGIN-CHEATSHEET-eventIsComplete
        event.isComplete();
        // END-CHEATSHEET-eventIsComplete

        // BEGIN-CHEATSHEET-waitEvent
        onHost::wait(event);
        // END-CHEATSHEET-waitEvent

        {
            // BEGIN-CHEATSHEET-allocHostBuffer
            // Allocate memory for the alpaka buffer, which is a dynamic 3-dimensional array
            // Memory allocations support any dimensionality
            concepts::IBuffer auto hostBuffer = onHost::allocHost<DataType>(extent3D);
            // END-CHEATSHEET-allocHostBuffer

            unused(hostBuffer);
        }

        {
            DataType* externPtr = nullptr;
            // BEGIN-CHEATSHEET-makeViewFromPtr
            auto extent = Vec{numElements};
            DataType* ptr = externPtr;
            concepts::IView auto hostView = makeView(api::host, ptr, extent);
            // END-CHEATSHEET-makeViewFromPtr

            unused(hostView);
        }

        {
            // BEGIN-CHEATSHEET-makeViewFromStdVector
            std::vector vec = std::vector<DataType>(42u);
            // the api is not required, std::vector is assumed to be api::host
            // a non owning view us usable within a kernel and on the host therefore no namespace 'onHost' is required
            auto hostView = makeView(vec);
            // END-CHEATSHEET-makeViewFromStdVector

            unused(hostView);
        }

        {
            // BEGIN-CHEATSHEET-makeViewStdArray
            std::array array = std::array<DataType, 2>{42u, 23};
            // call within host code: api::host is automatically assumed
            concepts::IView auto hostView = makeView(array);
            // call from within a cuda kernel: api::cuda is automatically assumed
            concepts::IView auto deviceView = makeView(array);
            // END-CHEATSHEET-makeViewStdArray

            unused(hostView, deviceView);
        }
        {
            // BEGIN-CHEATSHEET-allocLike
            // This allocLike + memcpy pattern is not specific to std::vector; it also works with std::array
            // and with alpaka Buffers/Views.
            // Construct a host container (here: std::vector) with arbitrary values.
            std::vector vec(42u, DataType{10});
            // Create a one-dimensional deviceBuffer with the same extent as 'vec'
            auto deviceBuffer = alpaka::onHost::allocLike(device, vec);
            // Copy host -> device directly from the vector into the allocated device buffer.
            // Note: if the queue is asynchronous, ensure the source memory container stays alive until the copy
            // completes.
            alpaka::onHost::memcpy(blockingQueue, deviceBuffer, vec);
            // END-CHEATSHEET-allocLike
            unused(vec, deviceBuffer);
        }
        {
            concepts::IBuffer auto buffer = onHost::allocHost<DataType>(numElements);
            buffer.destructorWaitFor(queue);
            // BEGIN-CHEATSHEET-dataPtr
            DataType* rawPtr = onHost::data(buffer);
            // END-CHEATSHEET-dataPtr

            unused(rawPtr);

            // BEGIN-CHEATSHEET-getPitches
            // number of bytes to the next element along the pitch dimension
            concepts::Vector auto bufferPitches = onHost::getPitches(buffer);
            // END-CHEATSHEET-getPitches

            unused(bufferPitches);

            // BEGIN-CHEATSHEET-initView
            // The buffer can have any dimensionality.
            // Memory manipulation functions supporting views too.
            // set all bytes to zero
            onHost::memset(queue, buffer, uint8_t{0});
            // element-wise fill with value
            onHost::fill(queue, buffer, 42);
            // END-CHEATSHEET-initView
        }

        concepts::Vector auto extentMd = Vec{value};
        {
            // BEGIN-CHEATSHEET-allocBuffer
            // the allocation is providing a shared buffer which will be
            // automatically freed if the last handle runs out of a life-time
            concepts::IBuffer auto devBuffer = onHost::alloc<DataType>(device, extentMd);
            // allocate memory which lives on the host but is accessible from the device too
            concepts::IBuffer auto devMappedBuffer = onHost::allocMapped<DataType>(device, extentMd);
            // allocate memory can be accessed from host and device (unified memory),
            // the real location depends on the native backend e.g. CUDA, OneApi, ...
            concepts::IBuffer auto devUnifiedBuffer = onHost::allocUnified<DataType>(device, extentMd);
            // allocate memory that is accessible after it is processed in the queue
            concepts::IBuffer auto devDeferredBuffer = onHost::allocDeferred<DataType>(queue, extentMd);
            // allocate memory accessible from host
            concepts::IBuffer auto hostBuffer = onHost::allocHost<DataType>(extentMd);
            // Data will not be automatically freed, user must take care that
            // the original data life-time is longer than the non-owning view.
            concepts::IView auto devNonOwningView = devBuffer.getView();
            // END-CHEATSHEET-allocBuffer

            unused(devMappedBuffer, devUnifiedBuffer, devNonOwningView, devDeferredBuffer);

            auto dstBuffer = devBuffer;
            auto srcBuffer = devMappedBuffer;

            // BEGIN-CHEATSHEET-memcpy
            // Memory manipulation functions supporting views too.
            onHost::memcpy(queue, dstBuffer, srcBuffer);
            // Providing the extent is optional and allow partial copies.
            onHost::memcpy(queue, dstBuffer, srcBuffer, extentMd);
            // END-CHEATSHEET-memcpy

            alpaka::onHost::wait(queue);
        }

        concepts::Vector auto numFramesMd = Vec{valueY, valueX};
        concepts::Vector auto frameExtentMd = Vec{1u, 1u};

        // BEGIN-CHEATSHEET-manualFrameSpec
        onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{numFramesMd, frameExtentMd};
        // END-CHEATSHEET-manualFrameSpec

        unused(frameSpec);

        {
            // BEGIN-CHEATSHEET-autoFrameSpec
            // Provides kernel start parameters sutable for the device and executor
            onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec(device, exec::anyExecutor, extentMd);
            // DataType is used to optimize the kernel parameters for working on data of this type
            onHost::concepts::FrameSpec auto simdFrameSpec
                = onHost::getSimdFrameSpec<DataType>(device, exec::anyExecutor, extentMd);
            // END-CHEATSHEET-autoFrameSpec

            unused(frameSpec, simdFrameSpec);
        }

        {
            int argumentsForConstructor = 42;

            // BEGIN-CHEATSHEET-createKernelWithArg
            Kernel kernel{argumentsForConstructor};
            // END-CHEATSHEET-createKernelWithArg

            auto foo = [=](auto const&... kernelArgs)
            {
                // BEGIN-CHEATSHEET-enqueueKernel
                // automatically deduct a fast executor for the given device
                queue.enqueue(frameSpec, KernelBundle{kernel, kernelArgs...});
                // or use a specific executor
                auto executor = exec::cpuSerial;
                queue.enqueue(
                    onHost::FrameSpec{frameSpec.getNumFrames(), frameSpec.getFrameExtents(), executor},
                    KernelBundle{kernel, kernelArgs...});
                // END-CHEATSHEET-enqueueKernel
            };
            unused(foo);
        }

        // Enqueue the memory fence kernel to ensure it compiles and links
        queue.enqueue(frameSpec, KernelBundle{MemoryFenceKernel{}});
    }
    catch(...)

    {
        // we do not want to exit in case of an error because the code is for the cheat sheet only
    }
}
