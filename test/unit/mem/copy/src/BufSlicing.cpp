/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Jakob Krude
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/Iterator.hpp>

#include <catch2/catch.hpp>

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4127) // suppress warning for c++17 conditional expression is constant
#endif

template<typename TDim, typename TIdx, typename TAcc, typename TData, typename TVec = alpaka::Vec<TDim, TIdx>>
struct TestContainer
{
    using AccQueueProperty = alpaka::Blocking;
    using DevQueue = alpaka::Queue<TAcc, AccQueueProperty>;
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;

    using DevHost = alpaka::DevCpu;
    using PltfHost = alpaka::Pltf<DevHost>;

    using BufHost = alpaka::Buf<DevHost, TData, TDim, TIdx>;
    using BufDevice = alpaka::Buf<DevAcc, TData, TDim, TIdx>;

    using SubView = alpaka::ViewSubView<DevAcc, TData, TDim, TIdx>;

    DevAcc const m_dev_acc;
    DevHost const m_dev_host;
    DevQueue m_dev_queue;


    // Constructor
    TestContainer()
        : m_dev_acc(alpaka::getDevByIdx<PltfAcc>(0u))
        , m_dev_host(alpaka::getDevByIdx<PltfHost>(0u))
        , m_dev_queue(m_dev_acc)
    {
    }


    auto create_host_buffer(TVec extents, bool indexed) -> BufHost
    {
        BufHost buf_host(alpaka::allocBuf<TData, TIdx>(m_dev_host, extents));
        if(indexed)
        {
            TData* const ptr = alpaka::getPtrNative(buf_host);
            for(TIdx i(0); i < extents.prod(); ++i)
            {
                ptr[i] = static_cast<TData>(i);
            }
        }
        return buf_host;
    }


    auto create_device_buffer(TVec extents) -> BufDevice
    {
        BufDevice buf_device(alpaka::allocBuf<TData, TIdx>(m_dev_acc, extents));
        return buf_device;
    }


    auto copy_to_acc(BufHost buf_host, BufDevice buf_acc, TVec extents) -> void
    {
        alpaka::memcpy(m_dev_queue, buf_acc, buf_host, extents);
    }


    auto copy_to_host(BufDevice buf_acc, BufHost buf_host, TVec extents) -> void
    {
        alpaka::memcpy(m_dev_queue, buf_host, buf_acc, extents);
    }


    auto slice_on_device(BufDevice buffer_to_be_sliced, TVec sub_view_extents, TVec offsets) -> BufDevice
    {
        BufDevice sliced_buffer = create_device_buffer(sub_view_extents);
        // Create a subView with a possible offset.
        SubView sub_view = SubView(buffer_to_be_sliced, sub_view_extents, offsets);
        // Copy the subView into a new buffer.
        alpaka::memcpy(m_dev_queue, sliced_buffer, sub_view, sub_view_extents);
        return sliced_buffer;
    }


    auto compare_buffer(BufHost const& buffer_a, BufHost const& buffer_b, TVec const& extents) const
    {
        TData const* const ptr_a = alpaka::getPtrNative(buffer_a);
        TData const* const ptr_b = alpaka::getPtrNative(buffer_b);
        for(TIdx i(0); i < extents.prod(); ++i)
        {
            INFO("Dim: " << TDim::value)
            INFO("Idx: " << typeid(TIdx).name())
            INFO("Acc: " << alpaka::traits::GetAccName<TAcc>::getAccName())
            INFO("i: " << i)
            REQUIRE(ptr_a[i] == Approx(ptr_b[i]));
        }
    }
};

using DataTypes = std::tuple<int, float, double>;

using TestAccWithDataTypes = alpaka::meta::CartesianProduct<std::tuple, alpaka::test::TestAccs, DataTypes>;

TEMPLATE_LIST_TEST_CASE("memBufSlicingTest", "[memBuf]", TestAccWithDataTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using Data = std::tuple_element_t<1, TestType>;
    using Dim = alpaka::Dim<Acc>;
    // fourth-dimension is not supposed to be tested currently
    if(Dim::value == 4)
    {
        return;
    }
    using Idx = alpaka::Idx<Acc>;
    TestContainer<Dim, Idx, Acc, Data> slicing_test;

    auto const extents
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();

    auto const extents_sub_view
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentSubView>();
    auto const offsets
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForOffset>();

    // This is the initial buffer.
    auto const indexed_buffer = slicing_test.create_host_buffer(extents, true);
    // This buffer will hold the sliced-buffer when it was copied to the host.
    auto result_buffer = slicing_test.create_host_buffer(extents_sub_view, false);

    // Copy of the indexBuffer on the deviceSide.
    auto device_buffer = slicing_test.create_device_buffer(extents);

    // Start: Main-Test
    slicing_test.copy_to_acc(indexed_buffer, device_buffer, extents);

    auto sliced_buffer = slicing_test.slice_on_device(device_buffer, extents_sub_view, offsets);

    slicing_test.copy_to_host(sliced_buffer, result_buffer, extents_sub_view);

    auto correct_results = slicing_test.create_host_buffer(extents_sub_view, false);
    Data* ptr_native = alpaka::getPtrNative(correct_results);
    using Dim1 = alpaka::DimInt<1u>;

    for(Idx i(0); i < extents_sub_view.prod(); ++i)
    {
        auto mapped_to_nd = alpaka::mapIdx<Dim::value, Dim1::value>(alpaka::Vec<Dim1, Idx>(i), extents_sub_view);
        auto added_offset = mapped_to_nd + offsets;
        auto mapped_to1_d = alpaka::mapIdx<Dim1::value>(added_offset,
                                                      extents)[0]; // take the only element in the vector
        ptr_native[i] = static_cast<Data>(mapped_to1_d);
    }

    // resultBuffer will be compared with the manually computed results.
    slicing_test.compare_buffer(result_buffer, correct_results, extents_sub_view);
}

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
