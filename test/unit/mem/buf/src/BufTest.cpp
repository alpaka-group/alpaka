/**
 * \file
 * Copyright 2015-2017 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/Extent.hpp>

#include <type_traits>
#include <numeric>

//-----------------------------------------------------------------------------
template<
    typename TAcc>
static auto testBufferMutable(
    alpaka::vec::Vec<alpaka::dim::Dim<TAcc>, alpaka::idx::Idx<TAcc>> const & extent)
-> void
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Queue = alpaka::test::queue::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    Queue queue(dev);

    //-----------------------------------------------------------------------------
    // alpaka::mem::buf::alloc
    auto buf(alpaka::mem::buf::alloc<Elem, Idx>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::vec::Vec<Dim, Idx>::zeros());
    alpaka::test::mem::view::testViewImmutable<
        Elem>(
            buf,
            dev,
            extent,
            offset);

    //-----------------------------------------------------------------------------
    alpaka::test::mem::view::testViewMutable<
        TAcc>(
            queue,
            buf);
}

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    auto const extent(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));

    testBufferMutable<
        TAcc>(
            extent);
}
};

//-----------------------------------------------------------------------------
struct TestTemplateZero
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    auto const extent(alpaka::vec::Vec<Dim, Idx>::zeros());

    testBufferMutable<
        TAcc>(
            extent);
}
};


//-----------------------------------------------------------------------------
template<
    typename TAcc>
static auto testBufferImmutable(
    alpaka::vec::Vec<alpaka::dim::Dim<TAcc>, alpaka::idx::Idx<TAcc>> const & extent)
-> void
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

    //-----------------------------------------------------------------------------
    // alpaka::mem::buf::alloc
    auto const buf(alpaka::mem::buf::alloc<Elem, Idx>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::vec::Vec<Dim, Idx>::zeros());
    alpaka::test::mem::view::testViewImmutable<
        Elem>(
            buf,
            dev,
            extent,
            offset);
}

//-----------------------------------------------------------------------------
struct TestTemplateConst
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    auto const extent(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));

    testBufferImmutable<
        TAcc>(
            extent);
}
};

TEST_CASE( "memBufBasicTest", "[memBuf]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}

TEST_CASE( "memBufZeroSizeTest", "[memBuf]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateZero() );
}

TEST_CASE( "memBufConstTest", "[memBuf]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateConst() );
}
