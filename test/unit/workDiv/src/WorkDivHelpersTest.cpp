/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch.hpp>

namespace
{
    template<typename TAcc>
    auto get_work_div()
    {
        using Dev = alpaka::Dev<TAcc>;
        using Pltf = alpaka::Pltf<Dev>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
        auto const grid_thread_extent = alpaka::Vec<Dim, Idx>::all(10);
        auto const thread_element_extent = alpaka::Vec<Dim, Idx>::ones();
        auto const work_div = alpaka::getValidWorkDiv<TAcc>(
            dev,
            grid_thread_extent,
            thread_element_extent,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        return work_div;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("getValidWorkDiv", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    // Note: getValidWorkDiv() is called inside getWorkDiv
    auto const work_div = get_work_div<Acc>();
    alpaka::ignore_unused(work_div);
}

TEMPLATE_LIST_TEST_CASE("isValidWorkDiv", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;

    Dev dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const work_div = get_work_div<Acc>();
    // Test both overloads
    REQUIRE(alpaka::isValidWorkDiv(alpaka::getAccDevProps<Acc>(dev), work_div));
    REQUIRE(alpaka::isValidWorkDiv<Acc>(dev, work_div));
}
