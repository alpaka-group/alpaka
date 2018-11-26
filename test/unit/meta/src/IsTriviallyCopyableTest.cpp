/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/IsTriviallyCopyable.hpp>

#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <functional>
#if BOOST_LANG_CUDA
#include <nvfunctional>
#endif

struct A {
    long long a;
};

struct B {
    B(int x): b(x+2) {}
    B(B const&) = default;
    int b;
};

using TriviallyCopyableTypes = std::tuple<char, short, int, long, long long, float, double, A, B, int[], double[]/*, B[]*/, int const, A const/*, B const []*/
#if BOOST_LANG_CUDA
        ,nvstd::function<void(bool)>
#endif
    >;

//-----------------------------------------------------------------------------
struct TestTemplateTriviallyCopyable
{
    template< typename T >
    void operator()()
    {
        constexpr bool IsTriviallyCopyableResult =
            alpaka::meta::IsTriviallyCopyable<
                T
            >::value;

        constexpr bool IsTriviallyCopyableReference =
            true;

        static_assert(
            IsTriviallyCopyableReference == IsTriviallyCopyableResult,
            "alpaka::meta::IsTriviallyCopyable failed!");
    }
};

TEST_CASE( "isTriviallyCopyable", "[memView]")
{
    alpaka::meta::forEachType< TriviallyCopyableTypes >( TestTemplateTriviallyCopyable() );
}

struct C {
    C(){}
    C(C const&) {}
};

struct D {
    virtual ~D() = default;
};

using NonTriviallyCopyableTypes = std::tuple<C, D, std::function<void(bool)>>;

//-----------------------------------------------------------------------------
struct TestTemplateNotTriviallyCopyable
{
    template< typename T >
    void operator()()
    {
        constexpr bool IsTriviallyCopyableResult =
            alpaka::meta::IsTriviallyCopyable<
                T
            >::value;

        constexpr bool IsTriviallyCopyableReference =
            false;

        static_assert(
            IsTriviallyCopyableReference == IsTriviallyCopyableResult,
            "alpaka::meta::IsTriviallyCopyable failed!");
    }
};

TEST_CASE( "isNotTriviallyCopyable", "[memView]")
{
    alpaka::meta::forEachType< NonTriviallyCopyableTypes >( TestTemplateNotTriviallyCopyable() );
}
