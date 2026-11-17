/* Copyright 2023 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego, Andrea Bocci, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/api/api.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"

#include <functional>
#include <tuple>
#include <utility>

#pragma once

namespace alpaka::onHost
{
    /**! Execute a callable for each tuple entry.
     *
     * @attention: Execution is short-circuited and stops after the first error.
     *
     * @param callable Callable that can be invoked with each tuple entry and returns an execution status.
     *        A return value of zero (`EXIT_SUCCESS`) indicates success; any non-zero value indicates a failure.
     * @param tuple Tuple like list of entries used to invoke the callable.
     * @return The disjunction of all returned error codes. If false, the result is `EXIT_SUCCESS`;
     *          otherwise, at least a failure occurred.
     */
    template<template<typename...> class T_TupleLike, typename... T_Entries>
    inline int executeForEach(auto&& callable, T_TupleLike<T_Entries...> const& tuple)
    {
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply(
                   [=](auto const&... tupleEntry)
                   {
                       static_assert(
                           (std::same_as<ALPAKA_TYPEOF(callable(tupleEntry)), int> && ...),
                           "The callable must return 'int'.");
                       return (static_cast<bool>(callable(tupleEntry)) || ...);
                   },
                   tuple)
                       == false
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }

    /**! execute a callable for each device specification if there is a device available
     *
     * The function contains a runtime check if at least one device is available, if there is no device the callable
     * will not be executed. Not executed combinations will return EXIT_SUCCESS.
     *
     * @attention: Execution is short-circuited and stops after the first error.
     *
     * @param callable Callable that can be invoked with device specification and returns an execution status.
     *        A return value of zero (`EXIT_SUCCESS`) indicates success; any non-zero value indicates a failure.
     * @param tupleOfDeviceSpecs Tuple like list of device specifications used to invoke the callable.
     *         otherwise, at least one failure occurred.
     * @return The disjunction of all returned error codes. If false, the result is `EXIT_SUCCESS`;
     *          otherwise, at least a failure occurred.
     */
    template<alpaka::concepts::DeviceSpec... T_DeviceSpecs>
    inline int executeForEachIfHasDevice(auto&& callable, std::tuple<T_DeviceSpecs...> const& tupleOfDeviceSpecs)
    {
        auto exe = [=](auto const& devSpec)
        {
            auto devSelector = onHost::makeDeviceSelector(devSpec);
            if(devSelector.isAvailable())
            {
                return callable(devSpec);
            }
            return EXIT_SUCCESS;
        };
        return executeForEach(exe, tupleOfDeviceSpecs);
    }
} // namespace alpaka::onHost
