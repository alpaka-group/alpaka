/* Copyright 2022 Andrea Bocci, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// make sure the CPU_B_SEQ_T_SEQ backend is always available
#ifndef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#    define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

// fill an trivial type with std::memset
template<typename T>
constexpr auto memset_value(int c) -> T
{
    T t;
    std::memset(&t, c, sizeof(T));
    return t;
}

template<typename TAcc, typename TElem, typename TIdx, typename TQueue, typename TExtent>
auto allocAsyncBufIfSupported(TQueue const& queue, TExtent const& extent)
    -> alpaka::Buf<alpaka::Dev<TQueue>, TElem, alpaka::Dim<TExtent>, TIdx>
{
    if constexpr(alpaka::hasAsyncBufSupport<alpaka::Dev<TAcc>, alpaka::Dim<TExtent>>)
    {
        return alpaka::allocAsyncBuf<TElem, TIdx>(queue, extent);
    }
    else
    {
        return alpaka::allocBuf<TElem, TIdx>(alpaka::getDev(queue), extent);
    }

    ALPAKA_UNREACHABLE(alpaka::allocBuf<TElem, TIdx>(alpaka::getDev(queue), extent));
}

// 0- and 1- dimensional space
using Idx = std::size_t;
using Dim0D = alpaka::DimInt<0u>;
using Dim1D = alpaka::DimInt<1u>;
using Scalar = alpaka::Vec<Dim0D, Idx>;

// enabled accelerators with 1-dimensional kernel space
using TestAccs = alpaka::test::EnabledAccs<Dim1D, Idx>;

TEMPLATE_LIST_TEST_CASE("zeroDimBuffer", "[zeroDimBuffer]", TestAccs)
{
    using DeviceAcc = TestType;
    using Device = alpaka::Dev<DeviceAcc>;
    using DeviceQueue = alpaka::Queue<DeviceAcc, alpaka::NonBlocking>;

    using HostAcc = alpaka::AccCpuSerial<Dim1D, Idx>;
    using Host = alpaka::DevCpu;
    using HostQueue = alpaka::Queue<HostAcc, alpaka::Blocking>;

    // check that a Scalar extent has exactly 1 element
    Scalar scalar;
    INFO("Scalar size: " << scalar.prod())
    CHECK(scalar.prod() == 1u);

    // CPU host
    auto const host = alpaka::getDevByIdx<Host>(0u);
    INFO("Using alpaka accelerator: " << alpaka::getAccName<HostAcc>())
    HostQueue hostQueue(host);

    // host buffer
    auto h_buffer1 = alpaka::allocBuf<int, Idx>(host, Scalar{});
    INFO(
        "host buffer allocated at " << std::data(h_buffer1) << " with " << alpaka::getExtentProduct(h_buffer1)
                                    << " element(s)")

    // async host buffer
    auto h_buffer2 = allocAsyncBufIfSupported<HostAcc, int, Idx>(hostQueue, Scalar{});
    INFO(
        "second host buffer allocated at " << std::data(h_buffer2) << " with " << alpaka::getExtentProduct(h_buffer2)
                                           << " element(s)")

    // host-side buffer memset
    const int value1 = 42;
    const int expected1 = memset_value<int>(value1);
    INFO("host-side buffer memset")
    alpaka::memset(hostQueue, h_buffer1, value1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *h_buffer1);

    // host-side async buffer memset
    const int value2 = 99;
    const int expected2 = memset_value<int>(value2);
    INFO("host-side async buffer memset")
    alpaka::memset(hostQueue, h_buffer2, value2);
    alpaka::wait(hostQueue);
    CHECK(expected2 == *h_buffer2);

    // host-host copies
    INFO("buffer host-host copies")
    alpaka::memcpy(hostQueue, h_buffer2, h_buffer1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *h_buffer2);
    alpaka::memcpy(hostQueue, h_buffer1, h_buffer2);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *h_buffer1);

    // GPU device
    auto const device = alpaka::getDevByIdx<Device>(0u);
    INFO("Using alpaka accelerator: " << alpaka::getAccName<DeviceAcc>())
    DeviceQueue deviceQueue(device);

    // device buffer
    auto d_buffer1 = alpaka::allocBuf<int, Idx>(device, Scalar{});
    INFO(
        "device buffer allocated at " << std::data(d_buffer1) << " with " << alpaka::getExtentProduct(d_buffer1)
                                      << " element(s)")

    // async or second sync device buffer
    auto d_buffer2 = allocAsyncBufIfSupported<DeviceAcc, int, Idx>(deviceQueue, Scalar{});
    INFO(
        "second device buffer allocated at " << std::data(d_buffer2) << " with " << alpaka::getExtentProduct(d_buffer2)
                                             << " element(s)")

    // host-device copies
    INFO("host-device copies")
    alpaka::memcpy(deviceQueue, d_buffer1, h_buffer1);
    alpaka::memcpy(deviceQueue, d_buffer2, h_buffer2);

    // device-device copies
    INFO("device-device copies")
    alpaka::memcpy(deviceQueue, d_buffer1, d_buffer2);
    alpaka::memcpy(deviceQueue, d_buffer2, d_buffer1);

    // device-side buffer memset
    INFO("device-side buffer memset")
    alpaka::memset(deviceQueue, d_buffer1, value1);
    alpaka::memset(deviceQueue, d_buffer2, value2);

    // device-host copies
    INFO("device-host copies")
    alpaka::memcpy(deviceQueue, h_buffer1, d_buffer1);
    alpaka::memcpy(deviceQueue, h_buffer2, d_buffer2);

    alpaka::wait(deviceQueue);
    CHECK(expected1 == *h_buffer1);
    CHECK(expected2 == *h_buffer2);
}
