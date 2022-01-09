/* Copyright 2022 Andrea Bocci
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
constexpr T memset_value(int c)
{
    T t;
    std::memset(&t, c, sizeof(T));
    return t;
}

//! check if asynchronous (queue-ordered) memory buffers are supported by the given Accelerator
template<typename TAcc>
static constexpr auto isAsyncBufferSupported() -> bool
{
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCudaRt>)
    {
        return (BOOST_LANG_CUDA >= BOOST_VERSION_NUMBER(11, 2, 0)) && (alpaka::Dim<TAcc>::value == 1);
    }
    else
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevHipRt>)
    {
        return false;
    }
    else
#endif // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
        if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevOacc>)
    {
        return false;
    }
    else
#endif // ALPAKA_ACC_ANY_BT_OACC_ENABLED

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevOmp5>)
    {
        return false;
    }
    else
#endif // ALPAKA_ACC_ANY_BT_OMP5_ENABLED

        return true;
}

template<typename TAcc, typename TElem, typename TIdx, typename TQueue, typename TExtent>
auto allocAsyncBufIfSupported(TQueue const& queue, TExtent const& extent)
    -> alpaka::Buf<alpaka::Dev<TQueue>, TElem, alpaka::Dim<TExtent>, TIdx>
{
    if constexpr(isAsyncBufferSupported<TAcc>())
    {
        return alpaka::allocAsyncBuf<TElem, TIdx>(queue, extent);
    }
    else
    {
        return alpaka::allocBuf<TElem, TIdx>(alpaka::getDev(queue), extent);
    }
}

// 0- and 1- dimensional space
using Idx = std::size_t;
using Dim1D = alpaka::DimInt<1u>;
using Vec1D = alpaka::Vec<Dim1D, Idx>;

// enabled accelerators with 1-dimensional kernel space
using TestAccs = alpaka::test::EnabledAccs<Dim1D, Idx>;

TEMPLATE_LIST_TEST_CASE("hostOnlyAPI", "[hostOnlyAPI]", TestAccs)
{
    using DeviceAcc = TestType;
    using Device = alpaka::Dev<DeviceAcc>;
    using DeviceQueue = alpaka::Queue<DeviceAcc, alpaka::NonBlocking>;

    using HostAcc = alpaka::AccCpuSerial<Dim1D, Idx>;
    using Host = alpaka::DevCpu;
    using HostQueue = alpaka::Queue<HostAcc, alpaka::Blocking>;

    // CPU host
    auto const host = alpaka::getDevByIdx<Host>(0u);
    INFO("Using alpaka accelerator: " << alpaka::getAccName<HostAcc>())
    HostQueue hostQueue(host);

    // host buffer
    auto h_buffer1 = alpaka::allocBuf<int, Idx>(host, Vec1D{Idx{42}});
    INFO(
        "host buffer allocated at " << alpaka::getPtrNative(h_buffer1) << " with "
                                    << alpaka::extent::getExtentProduct(h_buffer1) << " element(s)")

    // async host buffer
    auto h_buffer2 = allocAsyncBufIfSupported<HostAcc, int, Idx>(hostQueue, Vec1D{Idx{42}});
    INFO(
        "second host buffer allocated at " << alpaka::getPtrNative(h_buffer2) << " with "
                                           << alpaka::extent::getExtentProduct(h_buffer2) << " element(s)")

    // host-side memset
    const int value1 = 42;
    const int expected1 = memset_value<int>(value1);
    INFO("host-side memset")
    alpaka::memset(hostQueue, h_buffer1, value1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer1));

    // host-side async memset
    const int value2 = 99;
    const int expected2 = memset_value<int>(value2);
    INFO("host-side async memset")
    alpaka::memset(hostQueue, h_buffer2, value2);
    alpaka::wait(hostQueue);
    CHECK(expected2 == *alpaka::getPtrNative(h_buffer2));

    // host-host copies
    INFO("buffer host-host copies")
    alpaka::memcpy(hostQueue, h_buffer2, h_buffer1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer2));
    alpaka::memcpy(hostQueue, h_buffer1, h_buffer2);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer1));

    // GPU device
    auto const device = alpaka::getDevByIdx<Device>(0u);
    INFO("Using alpaka accelerator: " << alpaka::getAccName<DeviceAcc>())
    DeviceQueue deviceQueue(device);

    // device buffer
    auto d_buffer1 = alpaka::allocBuf<int, Idx>(device, Vec1D{Idx{42}});
    INFO(
        "device buffer allocated at " << alpaka::getPtrNative(d_buffer1) << " with "
                                      << alpaka::extent::getExtentProduct(d_buffer1) << " element(s)")

    // async or second sync device buffer
    auto d_buffer2 = allocAsyncBufIfSupported<DeviceAcc, int, Idx>(deviceQueue, Vec1D{Idx{42}});
    INFO(
        "second device buffer allocated at " << alpaka::getPtrNative(d_buffer2) << " with "
                                             << alpaka::extent::getExtentProduct(d_buffer2) << " element(s)")

    // host-device copies
    INFO("host-device copies")
    alpaka::memcpy(deviceQueue, d_buffer1, h_buffer1);
    alpaka::memcpy(deviceQueue, d_buffer2, h_buffer2);

    // device-device copies
    INFO("device-device copies")
    alpaka::memcpy(deviceQueue, d_buffer1, d_buffer2);
    alpaka::memcpy(deviceQueue, d_buffer2, d_buffer1);

    // device-side memset
    INFO("device-side memset")
    alpaka::memset(deviceQueue, d_buffer1, value1);
    alpaka::memset(deviceQueue, d_buffer2, value2);

    // device-host copies
    INFO("device-host copies")
    alpaka::memcpy(deviceQueue, h_buffer1, d_buffer1);
    alpaka::memcpy(deviceQueue, h_buffer2, d_buffer2);

    alpaka::wait(deviceQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer1));
    CHECK(expected2 == *alpaka::getPtrNative(h_buffer2));
}
