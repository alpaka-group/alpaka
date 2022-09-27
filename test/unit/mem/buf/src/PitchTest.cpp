/* Copyright 2022 Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <boost/mp11.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    using Idx = int;

    template<typename Dev, typename Dim>
    void checkBufferPitches(const Dev& dev, alpaka::Vec<Dim, Idx> extent, alpaka::Vec<Dim, Idx> expectedPitches)
    {
        auto buf = alpaka::allocBuf<float, Idx>(dev, extent);
        CAPTURE(extent);
        const auto pitches = alpaka::getPitchBytesVec(buf);
        CHECK(pitches == expectedPitches);
        boost::mp11::mp_for_each<boost::mp11::mp_iota<Dim>>(
            [&](auto ic)
            {
                constexpr auto i = decltype(ic)::value;
                CHECK(alpaka::getPitchBytes<i>(buf) == pitches[i]);
            });
    }
} // namespace

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
TEST_CASE("memBufPitchTest.AccCpuSerial", "[memBuf]")
{
    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<0>, Idx>;
    const auto dev = alpaka::getDevByIdx<alpaka::Pltf<alpaka::Dev<Acc>>>(0);

    checkBufferPitches(dev, alpaka::Vec<alpaka::DimInt<0>, Idx>{}, alpaka::Vec<alpaka::DimInt<0>, Idx>{});

    checkBufferPitches(dev, alpaka::Vec{10}, alpaka::Vec{40});

    checkBufferPitches(dev, alpaka::Vec{42, 10}, alpaka::Vec{1680, 40});
    checkBufferPitches(dev, alpaka::Vec{10, 42}, alpaka::Vec{1680, 168});

    checkBufferPitches(dev, alpaka::Vec{42, 10, 2}, alpaka::Vec{3360, 80, 8});
    checkBufferPitches(dev, alpaka::Vec{2, 10, 42}, alpaka::Vec{3360, 1680, 168});
}
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
TEST_CASE("memBufPitchTest.AccGpuCudaRt", "[memBuf]")
{
    using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<0>, Idx>;

    const auto dev = alpaka::getDevByIdx<alpaka::Pltf<alpaka::Dev<Acc>>>(0);

    checkBufferPitches(dev, alpaka::Vec<alpaka::DimInt<0>, Idx>{}, alpaka::Vec<alpaka::DimInt<0>, Idx>{});

    checkBufferPitches(dev, alpaka::Vec{10}, alpaka::Vec{40});

    constexpr auto pitch = 512;

    checkBufferPitches(dev, alpaka::Vec{42, 10}, alpaka::Vec{42 * pitch, pitch});
    checkBufferPitches(dev, alpaka::Vec{10, 42}, alpaka::Vec{10 * pitch, pitch});

    checkBufferPitches(dev, alpaka::Vec{42, 10, 2}, alpaka::Vec{42 * 10 * pitch, 10 * pitch, pitch});
    checkBufferPitches(dev, alpaka::Vec{2, 10, 42}, alpaka::Vec{2 * 10 * pitch, 10 * pitch, pitch});
}
#endif
