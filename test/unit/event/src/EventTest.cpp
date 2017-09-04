/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/stream/Stream.hpp>            // alpaka::test::stream::TestStreams
#include <alpaka/test/stream/StreamTestFixture.hpp> // alpaka::test::stream::StreamTestFixture

#include <boost/predef.h>                           // BOOST_COMP_CLANG
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(event)

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventTestShouldInitiallyBeTrue,
    TDevStream,
    alpaka::test::stream::TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    Fixture f;

    alpaka::event::Event< typename Fixture::Stream > event(f.m_dev);

    BOOST_REQUIRE_EQUAL(
        true,
        alpaka::event::test(event));
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventTestShouldBeTrueAfterEventProcessed,
    TDevStream,
    alpaka::test::stream::TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    Fixture f;

    alpaka::event::Event< typename Fixture::Stream > event(f.m_dev);

    alpaka::stream::enqueue(f.m_stream, event);

    // A synchronous stream will immediately process the event so no need to wait for it.
    if(!alpaka::test::stream::IsSyncStream<typename Fixture::Stream>::value)
    {
        alpaka::wait::wait(f.m_stream);
    }

    BOOST_REQUIRE_EQUAL(
        true,
        alpaka::event::test(event));
}

BOOST_AUTO_TEST_SUITE_END()
