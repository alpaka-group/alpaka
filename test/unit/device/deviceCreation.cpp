/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <type_traits>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/* A failed device creation must not modify the internal platform bookkeeping.
 *
 * In addition to the invalid index check this guards the exception safety of the device cache: the device handle
 * is only inserted into the platform after the device was constructed successfully. Therefore a user can catch the
 * exception and continue to use the platform, the existing device and a newly created platform handle.
 */
TEMPLATE_LIST_TEST_CASE("device creation failure keeps platform state valid", "[device][exception]", TestApis)
{
    auto const cfg = TestType::makeDict();
    auto const deviceSpec = onHost::makeDeviceSpec(cfg);
    auto selector = onHost::makeDeviceSelector(deviceSpec);
    UNSCOPED_INFO("DeviceSpec: " << getName(deviceSpec));
    if(!selector.isAvailable())
        SKIP("No device available for " << deviceSpec.getName());

    auto const deviceCount = selector.getDeviceCount();
    auto const device = selector.makeDevice(0);

    // an invalid device index must throw and must not modify the platform bookkeeping
    CHECK_THROWS_AS(selector.makeDevice(deviceCount), std::runtime_error);
    CHECK(selector.getDeviceCount() == deviceCount);
    CHECK(selector.makeDevice(0) == device);

    // a newly created platform handle must still be usable and return the cached device
    auto selectorFromNewPlatform = onHost::makeDeviceSelector(deviceSpec);
    CHECK(selectorFromNewPlatform.getDeviceCount() == deviceCount);
    CHECK(selectorFromNewPlatform.makeDevice(0) == device);
}
