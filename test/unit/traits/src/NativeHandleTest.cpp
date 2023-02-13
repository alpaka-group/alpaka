/* Copyright 2022 Antonio Di Pilato, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/traits/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("NativeHandle", "[handle]", alpaka::test::TestAccs)
{
    using Dev = alpaka::Dev<TestType>;
    using Pltf = alpaka::Pltf<Dev>;
    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto handle = alpaka::getNativeHandle(dev);

    STATIC_REQUIRE(std::is_same_v<alpaka::NativeHandle<Dev>, decltype(handle)>);
    STATIC_REQUIRE(std::is_same_v<alpaka::NativeHandle<Dev>, int>); // It won't work for SYCL backend
}
