/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/Concepts.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

struct ConceptExample
{
};
struct ConceptNonMatchingExample
{
};

struct ImplementerNotTagged
{
};

struct ImplementerNotTaggedButNonMatchingTagged
    : public alpaka::concepts::Implements<ConceptNonMatchingExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerTagged : public alpaka::concepts::Implements<ConceptExample, ImplementerTagged>
{
};

struct ImplementerTaggedButAlsoNonMatchingTagged
    : public alpaka::concepts::Implements<ConceptNonMatchingExample, ImplementerTaggedButAlsoNonMatchingTagged>
    , public alpaka::concepts::Implements<ConceptExample, ImplementerTaggedButAlsoNonMatchingTagged>
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
    , public alpaka::concepts::Implements<ConceptExample, ImplementerNotTagged>
{
};

struct ImplementerTaggedToBaseAlsoNonMatchingTagged
    : public ImplementerNotTaggedButNonMatchingTagged
    , public alpaka::concepts::Implements<ConceptExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerNonMatchingTaggedTaggedToBase
    : public ImplementerNotTagged
    , public alpaka::concepts::Implements<ConceptNonMatchingExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>
    , public alpaka::concepts::Implements<ConceptExample, ImplementerNotTagged>
{
};

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerNotTagged", "[core]")
{
    using ImplementationBase = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerNotTagged>;

    static_assert(
        std::is_same<ImplementerNotTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerNotTaggedButNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerNotTaggedButNonMatchingTagged>;

    static_assert(
        std::is_same<ImplementerNotTaggedButNonMatchingTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTagged", "[core]")
{
    using ImplementationBase = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerTagged>;

    static_assert(
        std::is_same<ImplementerTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTaggedButAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerTaggedButAlsoNonMatchingTagged>;

    static_assert(
        std::is_same<ImplementerTaggedButAlsoNonMatchingTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerWithTaggedBaseAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerWithTaggedBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same<ImplementerTaggedButAlsoNonMatchingTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerWithTaggedBase", "[core]")
{
    using ImplementationBase = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerWithTaggedBase>;

    static_assert(
        std::is_same<ImplementerTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTaggedToBase", "[core]")
{
    using ImplementationBase = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerTaggedToBase>;

    static_assert(
        std::is_same<ImplementerNotTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerTaggedToBaseAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same<ImplementerNotTaggedButNonMatchingTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("ImplementerNonMatchingTaggedTaggedToBase", "[core]")
{
    using ImplementationBase
        = alpaka::concepts::ImplementationBase<ConceptExample, ImplementerNonMatchingTaggedTaggedToBase>;

    static_assert(
        std::is_same<ImplementerNotTagged, ImplementationBase>::value,
        "alpaka::concepts::ImplementationBase failed!");
}
