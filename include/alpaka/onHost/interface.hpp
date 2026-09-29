/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/trait.hpp"

/** Functionality which is usable on the host CPU controller thread. */
namespace alpaka::onHost
{
    /** @{
     * @name Query extents
     */
    /** Object extents
     *
     * @param any can be a std::vector, std::array, ...
     * @return the extents of the object
     */
    inline decltype(auto) getExtents(auto&& any)
    {
        return internal::getExtents(ALPAKA_FORWARD(any));
    }

    /** Handle extents
     *
     * @param handle can be a view, a data
     * @return the extents of the object
     */
    inline decltype(auto) getExtents(alpaka::concepts::HasGet auto&& handle)
    {
        return internal::getExtents(*handle.get());
    }

    /** @} */

    /** @{
     * @name Query multi-dimensional pitches
     */
    /** Object pitches
     *
     * @param any can be a std::vector, std::array, ...
     * @return Multidimensional value with number of bytes to jump to the next value within the corresponding
     *         dimension.
     *         The inner-most dimension (x) is sizeof(value_type), the next dimension (y) is the byte-stride of a
     *         full row including padding, and so on for higher dimensions.
     *         Given an ND index, the element-wise product with the pitches summed (dot product) yields the byte
     *         offset from the start of the buffer to that data element.
     */
    inline decltype(auto) getPitches(auto&& any)
    {
        return internal::getPitches(ALPAKA_FORWARD(any));
    }

    /** Handle pitches
     *
     * @param handle can be a view, a data
     * @return Multidimensional value with number of bytes to jump to the next value within the corresponding
     *         dimension.
     *         The inner-most dimension (x) is sizeof(value_type), the next dimension (y) is the byte-stride of a
     *         full row including padding, and so on for higher dimensions.
     *         Given an ND index, the element-wise product with the pitches summed (dot product) yields the byte
     *         offset from the start of the buffer to that data element.
     */
    inline decltype(auto) getPitches(alpaka::concepts::HasGet auto&& handle)
    {
        return internal::getPitches(*handle.get());
    }

    /** @} */

    /** @{
     * @name Query the name
     */

    /** Compile-time available name for a given object.
     *
     * @param any object whose name shall be queried
     * @return a `std::string`-compatible value holding the static name
     */
    inline std::convertible_to<std::string> auto getStaticName(auto const& any)
    {
        return alpaka::internal::GetStaticName::Op<ALPAKA_TYPEOF(any)>{}(any);
    }

    /** Compile-time available name of an handle
     *
     * @param handle object whose name shall be queried
     * @return a `std::string`-compatible value holding the static name
     */
    inline std::convertible_to<std::string> auto getStaticName(concepts::StaticNameHandle auto const& handle)
    {
        return alpaka::internal::GetStaticName::Op<std::decay_t<decltype(*handle.get())>>{}(*handle.get());
    }

    /** Runtime name for a given object.
     *
     * @param any object whose name shall be queried
     * @return a `std::string`-compatible value holding the name
     */
    inline std::convertible_to<std::string> auto getName(auto&& any)
    {
        return alpaka::internal::GetName::Op<ALPAKA_TYPEOF(any)>{}(ALPAKA_FORWARD(any));
    }

    /** Runtime name for a given handle.
     *
     * @param handle object whose name shall be queried
     * @return a `std::string`-compatible value holding the name
     */
    inline std::convertible_to<std::string> auto getName(concepts::NameHandle auto const& handle)
    {
        return alpaka::internal::GetName::Op<std::decay_t<decltype(*handle.get())>>{}(*handle.get());
    }

    /** @} */

    /** Get the native handle of an handle.
     *
     * The native handle can be passed to the underlying backend API
     * (e.g. CUDA, HIP, OpenMP) for low-level operations.
     *
     * @param handle object exposing a native handle
     * @return the native handle returned by the backend-specific implementation
     */
    inline auto getNativeHandle(alpaka::concepts::HasGet auto&& handle)
    {
        return internal::getNativeHandle(*handle.get());
    }

    /** Get the native handle.
     *
     * The native handle can be passed to the underlying backend API
     * (e.g. CUDA, HIP, OpenMP) for low-level operations.
     *
     * @param any object providing a native handle
     * @return the native handle returned by the backend-specific implementation
     */
    inline auto getNativeHandle(auto&& any)
    {
        return internal::getNativeHandle(ALPAKA_FORWARD(any));
    }

    /** wait for all work to be finished
     *
     * Waits until all work submitted to any before this call has finished
     *
     * @param handle queue/device/event
     */
    inline void wait(alpaka::concepts::HasGet auto&& handle)
    {
        return internal::wait(*handle.get());
    }

    /** @{
     * @name Query raw pointer
     */
    /** pointer to data of an object
     *
     * For multi-dimensional data the data is not required to be continuous.
     *
     * @param any object providing data access (e.g. std::vector)
     * @return raw pointer to the underlying data (equivalent to `std::data`)
     */
    inline decltype(auto) data(auto&& any)
    {
        return internal::Data::data(ALPAKA_FORWARD(any));
    }

    /** pointer to data of an handle
     *
     * For multi-dimensional data the data is not required to be continuous.
     *
     * @param handle handle providing data access (e.g. view)
     * @return raw pointer to the underlying data
     */
    inline decltype(auto) data(alpaka::concepts::HasGet auto&& handle)
    {
        return internal::Data::data(*handle.get());
    }

    /** @} */

    /** @{
     * @name Host allocations
     */
    /** Allocate host memory for a given element type and extents.
     *
     * The allocation is performed on the host controller device
     * (`api::host` ans `deviceKind::cpu`).
     * The returned view owns the allocated memory.
     *
     * @tparam T_ValueType type of the data elements
     * @param extents number of elements per dimension (vector or scalar)
     * @return a view owning the newly allocated memory
     */
    template<typename T_ValueType>
    inline auto allocHost(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        auto device = makeHostDevice<T_ValueType>();
        Vec const extentsVec = extents;
        return internal::Alloc::Op<T_ValueType, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** Allocate host memory with the same value type and extents as an existing view.
     *
     * The content of the source view is **not** copied. The function deduces the
     * element type and extents from `view` and creates a new shared buffer on the
     * host controller device.
     *
     * @param view a view (e.g. `std::vector`, `std::array`, or any compatible type)
     * @return a view owning the newly allocated memory
     */
    inline auto allocHostLike(auto const& view)
    {
        auto device = makeHostDevice<ALPAKA_TYPEOF(view)>();
        return alloc<alpaka::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, internal::getExtents(view));
    }

    /** @} */

    /** @{
     * @name Device selection utilities
     */
    /** Resolve the list of executors supported for a device specification.
     *
     * This helper is used internally to build backend dictionaries.
     *
     * @param deviceSpec      device specification to be used
     * @param listOfExecutors tuple of executor types to be filtered
     * @return a tuple containing the supported executor types
     *
     * @{
     */
    constexpr auto getExecutorsList(auto const deviceSpec, auto const listOfExecutors)
        requires(ALPAKA_TYPEOF(deviceSpec)::isValid())
    {
        using DevSelectorType = decltype(makeDeviceSelector(deviceSpec));
        using DeviceType = decltype(std::declval<DevSelectorType>().makeDevice(0));
        using ExecutorListType = decltype(supportedExecutors(std::declval<DeviceType>(), listOfExecutors));
        return ExecutorListType{};
    }

    constexpr auto getExecutorsList(auto const deviceSpec, auto const listOfExecutors)
    {
        alpaka::unused(deviceSpec, listOfExecutors);
        return std::tuple<>{};
    }

    /**@} */

    /** Create a tuple of device specifications for a single API.
     *
     * Each device specifications combines the supplied API with one of the supported
     * device types for that API.
     *
     * @param api a single alpaka API (e.g. `api::cuda`, `api::hip`)
     * @return a tuple containing all device specifications for the given API
     */
    constexpr auto getDeviceSpecsFor(auto const api)
    {
        return std::apply(
            [api](auto... devType) constexpr { return std::make_tuple(DeviceSpec{api, devType}...); },
            supportedDevices(api));
    }

    /** Create a flattened tuple of device specification objects for a list of APIs.
     *
     * @param apiList a `std::tuple` containing the APIs
     * @return a tuple containing all device specifications for the given API
     */
    template<alpaka::concepts::Api... T_Apis>
    constexpr auto getDeviceSpecsFor(std::tuple<T_Apis...> const apiList)
    {
        return std::apply([](auto... api) constexpr { return std::tuple_cat(getDeviceSpecsFor(api)...); }, apiList);
    }

    /** Build a tuple of backends for a single device specification.
     *
     * A backend is the combination of a device specification and an executor.
     * Query its executor via `alpaka::getExecutor(backend)` and recover its device
     * specification via `alpaka::onHost::DeviceSpec{backend}`.
     *
     * @param deviceSpec the device specification to associate with the executors
     * @param listOfExecutors tuple of executor types
     * @return a tuple of backend objects, one per executor
     */
    constexpr auto createBackendsFor(auto const deviceSpec, auto const listOfExecutors)
    {
        return std::apply(
            [deviceSpec](auto... executor) constexpr
            {
                return std::make_tuple(
                    Dict{
                        DictEntry{object::api, getApi(deviceSpec)},
                        DictEntry{object::deviceKind, getDeviceKind(deviceSpec)},
                        DictEntry{object::exec, executor}}...);
            },
            listOfExecutors);
    }

    /** Create the complete backend list for all device specifications and executors.
     *
     * @param devSpecList tuple of device specifications
     * @param listOfExecutors tuple of executor types
     * @return a tuple of backend objects, for all executors
     */
    constexpr auto createBackendList(auto const devSpecList, auto const listOfExecutors)
    {
        return std::apply(
            [listOfExecutors](auto... devSpec) constexpr
            { return std::tuple_cat(createBackendsFor(devSpec, getExecutorsList(devSpec, listOfExecutors))...); },
            devSpecList);
    }

    /** Generate the full set of backend dictionaries for a set of APIs.
     *
     * The result contains a backend entry for each combination of supported device
     * specification for the APIs and executors.
     *
     * @param usedApis tuple of alpaka APIs to consider
     * @param listOfExecutors tuple of executor types
     * @return a tuple of backend dictionaries covering all APIs and executors
     */
    template<alpaka::concepts::Api... T_Apis>
    consteval auto allBackends(std::tuple<T_Apis...> const& usedApis, auto const listOfExecutors)
    {
        return std::apply(
            [listOfExecutors](auto... api) constexpr
            { return std::tuple_cat(createBackendList(getDeviceSpecsFor(api), listOfExecutors)...); },
            usedApis);
    }

    /** Generate the full set of backend dictionaries for a set of device kinds.
     *
     * The result contains a backend entry for each combination of supported device
     * specification and executors.
     *
     * @param usedDeviceSpecs tuple of alpaka device kinds
     * @param listOfExecutors tuple of executor types
     * @return a tuple of backend dictionaries covering all device kinds and executors
     */
    template<alpaka::concepts::DeviceSpec... T_DevicesSpecs>
    consteval auto allBackends(std::tuple<T_DevicesSpecs...> const& usedDeviceSpecs, auto const& listOfExecutors)
    {
        return std::tuple_cat(createBackendList(usedDeviceSpecs, listOfExecutors));
    }
} // namespace alpaka::onHost
