/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch.hpp>

#include <type_traits>
#include <numeric>

//-----------------------------------------------------------------------------
template<unsigned TDim, unsigned TElemsPerDim, typename TData>
struct TestTemplateCopy
{
template< typename TAcc >
auto operator()() const -> void
{
    using Dim = alpaka::dim::DimInt<TDim>;
    using Idx = std::size_t;
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using PltfHost = alpaka::pltf::PltfCpu;
    using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;

    auto const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    auto const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    QueueAcc devQueue{devAcc};

    constexpr Idx nElementsPerDim = TElemsPerDim;
    using Data = TData;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    const Vec extents{Vec::all(static_cast<Idx>(nElementsPerDim))};

    // Allocate host memory buffers
    //
    // The `alloc` method returns a reference counted buffer handle.
    // When the last such handle is destroyed, the memory is freed automatically.
    auto hostBuffer1(alpaka::mem::buf::alloc<Data, Idx>(devHost, extents));
    auto hostBuffer2(alpaka::mem::buf::alloc<Data, Idx>(devHost, extents));
    auto hostBuffer3(alpaka::mem::buf::alloc<Data, Idx>(devHost, extents));

    // Allocate accelerator memory buffers
    //
    // The interface to allocate a buffer is the same on the host and on the device.
    auto deviceBuffer1(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extents));
    auto deviceBuffer2(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extents));

    // fill initial host buffer
    Data * const pHostBuffer1{alpaka::mem::view::getPtrNative(hostBuffer1)};
    Data * const pHostBuffer3{alpaka::mem::view::getPtrNative(hostBuffer3)};

    for(Idx i{0}; i < extents.prod(); ++i)
    {
        pHostBuffer1[i] = static_cast<Data>(i+1);
        pHostBuffer3[i] = static_cast<Data>(0);
    }
    alpaka::mem::view::copy(devQueue, deviceBuffer1, hostBuffer1, extents);
    alpaka::mem::view::copy(devQueue, deviceBuffer2, deviceBuffer1, extents);
    alpaka::mem::view::copy(devQueue, hostBuffer2, deviceBuffer2, extents);
    alpaka::mem::view::copy(devQueue, hostBuffer3, hostBuffer2, extents);
    alpaka::wait::wait(devQueue);


    // This pointer can be used to directly write
    // some values into the buffer memory.
    // Mind, that only a host can write on host memory.
    // The same holds true for device memory.
    for(Idx i{0}; i < extents.prod(); ++i)
    {
        CHECK(pHostBuffer3[i] == static_cast<Data>(i+1));
    }
}
};
TEST_CASE( "viewCopy", "[memView]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt<1u>,
        std::size_t>;

    alpaka::meta::forEachType< TestAccs >(
      TestTemplateCopy<1u,3u,std::uint32_t>()
      );
    alpaka::meta::forEachType< TestAccs >(
      TestTemplateCopy<2u,3u,std::uint32_t>()
      );
    alpaka::meta::forEachType< TestAccs >(
      TestTemplateCopy<3u,3u,std::uint32_t>()
      );
}
