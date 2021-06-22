/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/meta/ForEachType.hpp>
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
        template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx>
        auto test_view_plain_ptr_immutable(
            alpaka::ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view,
            TDev const& dev,
            alpaka::Vec<TDim, TIdx> const& extent_view,
            alpaka::Vec<TDim, TIdx> const& offset_view) -> void
        {
            alpaka::test::testViewImmutable<TElem>(view, dev, extent_view, offset_view);
        }

        template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx>
        auto test_view_plain_ptr_mutable(
            alpaka::ViewPlainPtr<TDev, TElem, TDim, TIdx>& view,
            TDev const& dev,
            alpaka::Vec<TDim, TIdx> const& extent_view,
            alpaka::Vec<TDim, TIdx> const& offset_view) -> void
        {
            test_view_plain_ptr_immutable<TAcc>(view, dev, extent_view, offset_view);

            using Queue = alpaka::test::DefaultQueue<TDev>;
            Queue queue(dev);
            alpaka::test::testViewMutable<TAcc>(queue, view);
        }

        template<typename TAcc, typename TElem>
        auto test_view_plain_ptr() -> void
        {
            using Dev = alpaka::Dev<TAcc>;
            using Pltf = alpaka::Pltf<Dev>;

            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using View = alpaka::ViewPlainPtr<Dev, TElem, Dim, Idx>;

            Dev const dev = alpaka::getDevByIdx<Pltf>(0u);

            auto const extent_buf
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
            auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent_buf);

            auto const extent_view = extent_buf;
            auto const offset_view = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(0));
            View view(
                alpaka::getPtrNative(buf),
                alpaka::getDev(buf),
                alpaka::extent::getExtentVec(buf),
                alpaka::getPitchBytesVec(buf));

            alpaka::test::test_view_plain_ptr_mutable<TAcc>(view, dev, extent_view, offset_view);
        }

        template<typename TAcc, typename TElem>
        auto test_view_plain_ptr_const() -> void
        {
            using Dev = alpaka::Dev<TAcc>;
            using Pltf = alpaka::Pltf<Dev>;

            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using View = alpaka::ViewPlainPtr<Dev, TElem, Dim, Idx>;

            Dev const dev(alpaka::getDevByIdx<Pltf>(0u));

            auto const extent_buf
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
            auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent_buf);

            auto const extent_view = extent_buf;
            auto const offset_view = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(0));
            View const view(
                alpaka::getPtrNative(buf),
                alpaka::getDev(buf),
                alpaka::extent::getExtentVec(buf),
                alpaka::getPitchBytesVec(buf));

            alpaka::test::test_view_plain_ptr_immutable<TAcc>(view, dev, extent_view, offset_view);
        }

        template<typename TAcc, typename TElem>
        auto test_view_plain_ptr_operators() -> void
        {
            using Dev = alpaka::Dev<TAcc>;
            using Pltf = alpaka::Pltf<Dev>;

            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using View = alpaka::ViewPlainPtr<Dev, TElem, Dim, Idx>;

            Dev const dev = alpaka::getDevByIdx<Pltf>(0u);

            auto const extent_buf
                = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
            auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent_buf);

            View view(
                alpaka::getPtrNative(buf),
                alpaka::getDev(buf),
                alpaka::extent::getExtentVec(buf),
                alpaka::getPitchBytesVec(buf));

            // copy-constructor
            View view_copy(view);

            // move-constructor
            View view_move(std::move(view_copy));
        }
    } // namespace test
} // namespace alpaka
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

TEMPLATE_LIST_TEST_CASE("viewPlainPtrTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::test_view_plain_ptr<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewPlainPtrConstTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::test_view_plain_ptr_const<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewPlainPtrOperatorTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::test_view_plain_ptr_operators<TestType, float>();
}
