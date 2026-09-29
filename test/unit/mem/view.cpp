/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <vector>

using namespace alpaka;

using DeviceSpecs = std::decay_t<decltype(onHost::getDeviceSpecsFor(onHost::enabledApis))>;

TEMPLATE_LIST_TEST_CASE("makeView factories", "[mem][view]", DeviceSpecs)
{
    auto deviceSpec = TestType{};
    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        SKIP("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = selector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    auto queue = device.makeQueue(queueKind::blocking);
    auto host = onHost::makeHostDevice();
    concepts::Vector auto extents = Vec<size_t, 2u>{3, 5};

    auto checkRoundTrip = [&](auto const& source, auto expected)
    {
        auto deviceBuffer = onHost::alloc<int>(device, extents);
        auto result = onHost::allocHost<int>(extents);

        onHost::memcpy(queue, deviceBuffer, source);
        onHost::memcpy(queue, result, deviceBuffer);

        meta::ndLoopIncIdx(
            extents,
            [&](auto idx)
            {
                INFO("index: " << idx);
                REQUIRE(result[idx] == expected(idx));
            });
    };

    SECTION("pointer and extents")
    {
        std::vector<int> input(extents.y() * extents.x());
        for(size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<int>(100 + i);

        concepts::IView auto view = makeView(host, input.data(), extents);
        checkRoundTrip(view, [&](auto idx) { return input[idx.y() * extents.x() + idx.x()]; });
    }

    SECTION("pointer, extents, and pitches")
    {
        // Number of elements per row.
        constexpr size_t rowStrideElements = 8;
        std::vector<int> input(extents.y() * rowStrideElements, -1);

        for(size_t y = 0; y < extents.y(); ++y)
            for(size_t x = 0; x < extents.x(); ++x)
                input[y * rowStrideElements + x] = static_cast<int>(100 * y + x);

        /* Calculate byte pitches for the padded allocation, then expose only
         * the first five elements of each row through the view.
         */
        concepts::Vector auto bytePitches = calculatePitchesFromExtents<int>(Vec{extents.y(), rowStrideElements});
        concepts::IView auto view = makeView(host, input.data(), extents, bytePitches);

        checkRoundTrip(view, [&](auto idx) { return input[idx.y() * rowStrideElements + idx.x()]; });
    }

    SECTION("full self describing object")
    {
        auto input = onHost::allocHost<int>(extents);
        meta::ndLoopIncIdx(extents, [&](auto idx) { input[idx] = static_cast<int>(100 * idx.y() + idx.x()); });

        concepts::IView auto view = makeView(input);
        checkRoundTrip(view, [&](auto idx) { return static_cast<int>(100 * idx.y() + idx.x()); });
    }

    SECTION("explicit API and std::span")
    {
        std::vector<int> input(extents.y() * extents.x());
        for(size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<int>(200 + i);

        std::span<int> span{input};
        concepts::IView auto view = makeView(getApi(host), span);

        concepts::Vector auto spanExtents = Vec{input.size()};
        auto deviceBuffer = onHost::alloc<int>(device, spanExtents);
        auto result = onHost::allocHost<int>(spanExtents);

        onHost::memcpy(queue, deviceBuffer, view);
        onHost::memcpy(queue, result, deviceBuffer);

        for(size_t i = 0; i < input.size(); ++i)
            REQUIRE(result[Vec{i}] == input[i]);
    }
}
