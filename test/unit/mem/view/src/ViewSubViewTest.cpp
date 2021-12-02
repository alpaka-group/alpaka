/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <numeric>
#include <type_traits>

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'std::uint8_t*' to 'Elem*' increases required alignment of target type"
#endif

namespace alpaka
{
    namespace test
    {
        template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
        auto test_view_sub_view_immutable(
            alpaka::ViewSubView<TDev, TElem, TDim, TIdx> const& view,
            TBuf& buf,
            TDev const& dev,
            alpaka::Vec<TDim, TIdx> const& extent_view,
            alpaka::Vec<TDim, TIdx> const& offset_view) -> void
        {
            alpaka::test::testViewImmutable<TElem>(view, dev, extent_view, offset_view);

            // alpaka::traits::GetPitchBytes
            // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
            {
                auto const pitch_buf = alpaka::getPitchBytesVec(buf);
                auto const pitch_view = alpaka::getPitchBytesVec(view);

                for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
                {
                    REQUIRE(pitch_buf[i - static_cast<TIdx>(1u)] == pitch_view[i - static_cast<TIdx>(1u)]);
                }
            }

            // alpaka::traits::GetPtrNative
            // The native pointer has to be exactly the value we calculate here.
            {
                auto *view_ptr_native = reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(buf));
                auto const pitch_buf = alpaka::getPitchBytesVec(buf);
                for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
                {
                    auto const pitch
                        = (i < static_cast<TIdx>(TDim::value)) ? pitch_buf[i] : static_cast<TIdx>(sizeof(TElem));
                    view_ptr_native += offset_view[i - static_cast<TIdx>(1u)] * pitch;
                }
                REQUIRE(reinterpret_cast<TElem*>(view_ptr_native) == alpaka::getPtrNative(view));
            }
        }

        template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
        auto test_view_sub_view_mutable(
            alpaka::ViewSubView<TDev, TElem, TDim, TIdx>& view,
            TBuf& buf,
            TDev const& dev,
            alpaka::Vec<TDim, TIdx> const& extent_view,
            alpaka::Vec<TDim, TIdx> const& offset_view) -> void
        {
            test_view_sub_view_immutable<TAcc>(view, buf, dev, extent_view, offset_view);

            using Queue = alpaka::test::DefaultQueue<TDev>;
            Queue queue(dev);
            alpaka::test::testViewMutable<TAcc>(queue, view);
        }

        template<typename TAcc, typename TElem>
        auto test_view_sub_view_no_offset() -> void
        {
            using Dev = alpaka::Dev<TAcc>;
            using Pltf = alpaka::Pltf<Dev>;

            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

            Dev const dev = alpaka::getDevByIdx<Pltf>(0u);

            auto const extent_buf
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
            auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent_buf);

            auto const extent_view = extent_buf;
            auto const offset_view = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(0));
            View view(buf);

            alpaka::test::test_view_sub_view_mutable<TAcc>(view, buf, dev, extent_view, offset_view);
        }

        template<typename TAcc, typename TElem>
        auto test_view_sub_view_offset() -> void
        {
            using Dev = alpaka::Dev<TAcc>;
            using Pltf = alpaka::Pltf<Dev>;

            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

            Dev const dev = alpaka::getDevByIdx<Pltf>(0u);

            auto const extent_buf
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
            auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent_buf);

            auto const extent_view = alpaka::
                createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentSubView>();
            auto const offset_view
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForOffset>();
            View view(buf, extent_view, offset_view);

            alpaka::test::test_view_sub_view_mutable<TAcc>(view, buf, dev, extent_view, offset_view);
        }

        template<typename TAcc, typename TElem>
        auto test_view_sub_view_offset_const() -> void
        {
            using Dev = alpaka::Dev<TAcc>;
            using Pltf = alpaka::Pltf<Dev>;

            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

            Dev const dev = alpaka::getDevByIdx<Pltf>(0u);

            auto const extent_buf
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
            auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent_buf);

            auto const extent_view = alpaka::
                createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentSubView>();
            auto const offset_view
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForOffset>();
            View const view(buf, extent_view, offset_view);

            alpaka::test::test_view_sub_view_immutable<TAcc>(view, buf, dev, extent_view, offset_view);
        }
    } // namespace test
} // namespace alpaka
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

TEMPLATE_LIST_TEST_CASE("viewSubViewNoOffsetTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::test_view_sub_view_no_offset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::test_view_sub_view_offset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetConstTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::test_view_sub_view_offset_const<TestType, float>();
}
