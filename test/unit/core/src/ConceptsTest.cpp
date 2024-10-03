/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Concepts.hpp>

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

struct InterfaceExample
{
};

struct InterfaceNonMatchingExample
{
};

struct ImplementerNotTagged
{
};

struct ImplementerNotTaggedButNonMatchingTagged
    : public alpaka::interface::Implements<InterfaceNonMatchingExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerTagged : public alpaka::interface::Implements<InterfaceExample, ImplementerTagged>
{
};

struct ImplementerTaggedButAlsoNonMatchingTagged
    : public alpaka::interface::Implements<InterfaceNonMatchingExample, ImplementerTaggedButAlsoNonMatchingTagged>
    , public alpaka::interface::Implements<InterfaceExample, ImplementerTaggedButAlsoNonMatchingTagged>
{
};

struct ImplementerWithTaggedBase : public ImplementerTagged
{
};

struct ImplementerWithTaggedBaseAlsoNonMatchingTagged : public ImplementerTaggedButAlsoNonMatchingTagged
{
};

struct ImplementerTaggedToBase
    : public ImplementerNotTagged
    , public alpaka::interface::Implements<InterfaceExample, ImplementerNotTagged>
{
};

struct ImplementerTaggedToBaseAlsoNonMatchingTagged
    : public ImplementerNotTaggedButNonMatchingTagged
    , public alpaka::interface::Implements<InterfaceExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerNonMatchingTaggedTaggedToBase
    : public ImplementerNotTagged
    , public alpaka::interface::Implements<InterfaceNonMatchingExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>
    , public alpaka::interface::Implements<InterfaceExample, ImplementerNotTagged>
{
};

TEST_CASE("ImplementerNotTagged", "[core]")
{
    using ImplementationBase = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerNotTagged>;

    static_assert(
        std::is_same_v<ImplementerNotTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerNotTaggedButNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerNotTaggedButNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerNotTaggedButNonMatchingTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTagged", "[core]")
{
    using ImplementationBase = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerTagged>;

    static_assert(
        std::is_same_v<ImplementerTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTaggedButAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerTaggedButAlsoNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerTaggedButAlsoNonMatchingTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerWithTaggedBaseAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerWithTaggedBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerTaggedButAlsoNonMatchingTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerWithTaggedBase", "[core]")
{
    using ImplementationBase = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerWithTaggedBase>;

    static_assert(
        std::is_same_v<ImplementerTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTaggedToBase", "[core]")
{
    using ImplementationBase = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerTaggedToBase>;

    static_assert(
        std::is_same_v<ImplementerNotTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTaggedToBaseAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerNotTaggedButNonMatchingTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerNonMatchingTaggedTaggedToBase", "[core]")
{
    using ImplementationBase
        = alpaka::interface::ImplementationBase<InterfaceExample, ImplementerNonMatchingTaggedTaggedToBase>;

    static_assert(
        std::is_same_v<ImplementerNotTagged, ImplementationBase>,
        "alpaka::interface::ImplementationBase failed!");
}
