/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onHost/Event.hpp"
#include "alpaka/onHost/EventPolicyList.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

#include <bit>
#include <climits>

namespace alpaka::onHost
{
    /** @brief Description of a specific device that one can schedule kernels on.
     *
     * @details
     * A device is the combination of an alpaka::deviceKind::onHost::DeviceKind and an alpaka::concepts::Api,
     * representing an entity that one can schedule work on.
     *
     * @tparam T_Api The Api powering this device.
     * @tparam T_DeviceKind The kind of device it is.
     */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct Device
    {
    private:
        using PlatformHandle = ALPAKA_TYPEOF(internal::makePlatform(T_Api{}, T_DeviceKind{}));
        using DeviceHandle = ALPAKA_TYPEOF(
            internal::MakeDevice::Op<typename PlatformHandle::element_type>{}(
                *std::declval<PlatformHandle>().get(),
                0u));
        DeviceHandle m_device;

    public:
        friend struct alpaka::internal::GetName;
        friend struct internal::GetNativeHandle;

        using element_type = typename DeviceHandle::element_type;

        auto get() const
        {
            return m_device.get();
        }

        template<typename T_Device>
        Device(Handle<T_Device>&& internalDeviceHandle)
            : m_device{std::forward<Handle<T_Device>>(internalDeviceHandle)}
        {
        }

        void _()
        {
            static_assert(internal::concepts::Device<element_type>);
        }

        std::string getName() const
        {
            return alpaka::internal::GetName::Op<std::decay_t<decltype(*m_device.get())>>{}(*m_device.get());
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return internal::getNativeHandle(*m_device.get());
        }

        bool operator==(Device const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Device const& other) const
        {
            return this->get() != other.get();
        }

        /** Create a queue for this device.
         *
         * @attention If you call this method multiple times it is allowed that you always get the same handle
         * back. There is no guarantee that you will get independent queues.
         *
         * Enqueuing tasks into two different queues does not guarantee that these tasks run in parallel.
         * Running tasks from different tasks sequentially is valid behavior. Enqueuing into two individual queues only
         * signifies that the tasks are independent of each other and their order of execution is independent.
         *
         * By default, the queue is non-blocking and timing is disabled. It accepts synchronization-only events, which
         * do not provide timing information.
         *
         * @return A onHost::Queue that tasks and memory operations can be enqueued on.
         */
        auto makeQueue()
        {
            return makeQueue(QueuePolicyList{});
        }

        /** @copydoc makeQueue()
         *
         * @param firstPolicy First queue construction policy.
         * @param policies Additional queue construction policies.
         */
        auto makeQueue(alpaka::concepts::QueuePolicy auto firstPolicy, alpaka::concepts::QueuePolicy auto... policies)
        {
            return makeQueue(QueuePolicyList{firstPolicy, policies...});
        }

        /** @copydoc makeQueue()
         *
         * @param policies Queue construction policies. Supported policies are:
         *   - Blocking behaviour:
         *    - queueKind::nonBlocking: enqueue returns immediately; completion of the enqueued operation
         * must be ensured via onHost::wait(queue) or by enqueuing dependent operations onto the same queue.
         *    - queueKind::blocking: each enqueue only returns after the operation is complete and its effects are
         * host-visible.
         *   - timing::disabled or timing::enabled: whether the queue supports timing-enabled events.
         */
        auto makeQueue(alpaka::concepts::QueuePolicyList auto const& policies)
        {
            return Queue{
                internal::MakeQueue::Op<ALPAKA_TYPEOF(*m_device.get()), ALPAKA_TYPEOF(policies)>{}(
                    *m_device.get(),
                    policies),
                policies};
        }

        /** Create an event for this device.
         *
         * @return An event that can be used to synchronize multiple queues.
         */
        auto makeEvent()
        {
            return makeEvent(EventPolicyList{});
        }

        /** @copydoc makeEvent()
         *
         * @param firstPolicy First event construction policy.
         * @param policies Additional event construction policies.
         */
        auto makeEvent(alpaka::concepts::EventPolicy auto firstPolicy, alpaka::concepts::EventPolicy auto... policies)
        {
            return makeEvent(EventPolicyList{firstPolicy, policies...});
        }

        /** @copydoc makeEvent()
         *
         * @param policies Event construction policies. Supported policies are:
         *   - timing::disabled or timing::enabled: whether the event records timing information.
         */
        auto makeEvent(alpaka::concepts::EventPolicyList auto const& policies)
        {
            return Event{
                internal::MakeEvent::Op<ALPAKA_TYPEOF(*m_device.get()), ALPAKA_TYPEOF(policies)>{}(
                    *m_device.get(),
                    policies),
                policies};
        }

        /** Blocks the caller until the given handle executes all work
         */
        void wait()
        {
            return internal::wait(*m_device.get());
        }

        /** Properties of a given device
         *
         * @attention Currently only a handful of entries is available. The object will be refactored soon and will
         * become most likely a compile time dictionary tu support optional entries.
         */

        inline DeviceProperties getDeviceProperties() const
        {
            return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*m_device.get())>{}(*m_device.get());
        }

        size_t getFreeGlobalMemBytes() const
        {
            return internal::GetFreeGlobalMemBytes::Op<ALPAKA_TYPEOF(*m_device.get())>{}(*m_device.get());
        }

        constexpr alpaka::concepts::DeviceKind auto getDeviceKind() const
        {
            return T_DeviceKind{};
        }

        constexpr alpaka::concepts::Api auto getApi() const
        {
            return T_Api{};
        }
    };

    template<typename T_Device>
    Device(Handle<T_Device>&&) -> Device<
        ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Device>())),
        ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Device>()))>;

    /** @{
     * @name Device allocations
     */
    /** Allocate memory on the given device
     *
     * @tparam T_Type type of the data elements
     * @param device device handle
     * @param extents number of elements for each dimension
     * @return memory owning view to the allocated memory
     */
    template<typename T_Type>
    inline auto alloc(concepts::Device auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::Alloc::Op<T_Type, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** Allocate memory on the given device with unified virtual memory
     *
     * This memory can be accessed from all devices with the same Api and device kind. Depending on the backend e.g.
     * OneApi memory can be accessed by other device kind devices if they are using the same native context. It is not
     * allowed to access the data on two devices at the same time, this must be avoided by explicit synchronizations.
     * Unified memory follows the rules of UVM memory of the device backend e.g. CUDA, HIP, ...
     *
     * @tparam T_Type type of the data elements
     * @param device device handle
     * @param extents number of elements for each dimension
     * @return Managed view to the allocated memory
     */
    template<typename T_Type>
    inline auto allocUnified(concepts::Device auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocUnified::Op<T_Type, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** Allocates unified memory on the device associated with the given queue.
     *
     * This memory can be accessed from all devices with the same Api and device kind. Depending on the backend e.g.
     * OneApi memory can be accessed by other device kind devices if they are using the same native context. It is not
     * allowed to access the data on two devices at the same time, this must be avoided by explicit synchronizations.
     * Unified memory follows the rules of UVM memory of the device backend e.g. CUDA, HIP, ...
     *
     * @ingroup foo
     *
     * @tparam T_Type type of the data elements
     * @param queue queue handle
     * @param extents number of elements for each dimension
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline auto allocUnified(
        Queue<T_Device, T_Policies> const& queue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocUnified::
            Op<T_Type, std::decay_t<decltype(*queue.getDevice().get())>, ALPAKA_TYPEOF(extentsVec)>{}(
                *queue.getDevice().get(),
                extentsVec);
    }

    /** Allocate pinned memory on the host which is mapped into the address space of the device
     *
     * Mapped memory is located on the host and is transferred for each access via the PCIe/Nvlink bus. The performance
     * on the device is mostly pure. Mapped memory should be used for host memory if you transfer memory between host
     * and device via `onHost::memcpy()` because the transfer will be optimized for latency and performance.
     *
     * @tparam T_Type type of the data elements
     * @param device device handle
     * @param extents number of elements for each dimension
     */
    template<typename T_Type>
    inline auto allocMapped(concepts::Device auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocMapped::Op<T_Type, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** Allocate pinned memory on the host which is mapped into the address space of the device
     *
     * Mapped memory is located on the host and is transferred for each access via the PCIe/Nvlink bus. The performance
     * on the device is mostly pure. Mapped memory should be used for host memory if you transfer memory between host
     * and device via `onHost::memcpy()` because the transfer will be optimized for latency and performance.
     *
     * @tparam T_Type type of the data elements
     * @param queue queue handle
     * @param extents number of elements for each dimension
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline auto allocMapped(
        Queue<T_Device, T_Policies> const& queue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return allocMapped<T_Type>(queue.getDevice(), extents);
    }

    /** Allocate memory on the given device based on a view
     *
     * Derives type and extents of the memory from the view.
     * The content of the memory is NOT copied to the created allocated memory.
     *
     * @param device device handle
     * @param[in] view memory where properties will be derived from
     *
     * @return memory owning view to the allocated memory
     */
    inline auto allocLike(concepts::Device auto const& device, auto const& view)
    {
        return alloc<alpaka::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, internal::getExtents(view));
    }

    ///@}

    /** Check if the given view is accessible on the given device
     *
     * @param device device handle
     * @param view memory where properties will be derived from
     * @return true if the view is accessible on the device, false otherwise.
     * alpaka can not detect all memory access types therefore the result can be false even if the memory is accessible
     * because the view was allocated with a UVM allocator.
     *
     */
    inline bool isDataAccessible(concepts::Device auto const& device, alpaka::concepts::IView auto const& view)
    {
        return internal::IsDataAccessible::FirstPath<ALPAKA_TYPEOF(*device.get()), ALPAKA_TYPEOF(view)>{}(
                   *device.get(),
                   view)
               || internal::IsDataAccessible::SecondPath<
                   ALPAKA_TYPEOF(getApi(view)),
                   ALPAKA_TYPEOF(getDeviceKind(device)),
                   ALPAKA_TYPEOF(view)>{}(getApi(view), getDeviceKind(device), view);
    }

    /** Check if the given view is accessible on the device of the given queue
     *
     * @param queue queue handle
     */
    template<typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline bool isDataAccessible(Queue<T_Device, T_Policies> const& queue, alpaka::concepts::IView auto const& view)
    {
        return internal::IsDataAccessible::FirstPath<ALPAKA_TYPEOF(*queue.getDevice().get()), ALPAKA_TYPEOF(view)>{}(
                   *queue.getDevice().get(),
                   view)
               || internal::IsDataAccessible::SecondPath<
                   ALPAKA_TYPEOF(getApi(view)),
                   ALPAKA_TYPEOF(getDeviceKind(queue.getDevice())),
                   ALPAKA_TYPEOF(view)>{}(getApi(view), getDeviceKind(queue.getDevice()), view);
    }

    /** Provides a frame specification to operate on a given index range
     *
     * @param extents size of the index range
     * @return frame specification
     */
    template<typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    inline constexpr concepts::FrameSpec auto getFrameSpec(
        Device<T_Api, T_DeviceKind> const& device,
        alpaka::concepts::Executor auto executor,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        if constexpr(executor == exec::anyExecutor)
        {
            auto usedExecutor = defaultExecutor(device);
            return internal::getFrameSpec(*device.get(), usedExecutor, extents);
        }
        else
            return internal::getFrameSpec(*device.get(), executor, extents);
    }

    /** Provides a frame specification to operate on a given index range
     *
     * The frame specification will be optimized for SIMD executions in the highest dimension
     * for a flat non-hierarchical execution via onAcc::worker::threadsInGrid.
     * Do not use this functions for kernel using hierarchical thread parallelism, in many cases the frame
     * specification depends on the outer parallelism in the kernel.
     *
     * @tparam T_DataType the data type for which you would like to SIMD optimize
     * @param extents number of elements for each dimension of the type T_DataType
     * @return frame specification
     */
    template<typename T_DataType, typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    inline constexpr concepts::FrameSpec auto getSimdFrameSpec(
        Device<T_Api, T_DeviceKind> const& device,
        alpaka::concepts::Executor auto executor,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        if constexpr(executor == exec::anyExecutor)
        {
            auto usedExecutor = defaultExecutor(device);
            return internal::getSimdFrameSpec<T_DataType>(*device.get(), usedExecutor, extents);
        }
        else
            return internal::getSimdFrameSpec<T_DataType>(*device.get(), executor, extents);
    }
} // namespace alpaka::onHost
