/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'std::uint8_t*' to 'Elem*' increases required alignment of target type"
#endif

namespace alpaka::test
{
    template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
    auto testViewSubViewImmutable(
        alpaka::ViewSubView<TDev, TElem, TDim, TIdx> const& view,
        TBuf& buf,
        TDev const& dev,
        alpaka::Vec<TDim, TIdx> const& extentView,
        alpaka::Vec<TDim, TIdx> const& offsetView) -> void
    {
        alpaka::test::testViewImmutable<TElem>(view, dev, extentView, offsetView);

        // alpaka::trait::GetPitchBytes
        // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
        {
            auto const pitchBuf = alpaka::getPitchBytesVec(buf);
            auto const pitchView = alpaka::getPitchBytesVec(view);

            for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
            {
                REQUIRE(pitchBuf[i - static_cast<TIdx>(1u)] == pitchView[i - static_cast<TIdx>(1u)]);
            }
        }

        // alpaka::trait::GetPtrNative
        // The native pointer has to be exactly the value we calculate here.
        {
            auto viewPtrNative = reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(buf));
            if constexpr(TDim::value > 0)
            {
                auto const pitchBuf = alpaka::getPitchBytesVec(buf);
                for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
                {
                    auto const pitch
                        = (i < static_cast<TIdx>(TDim::value)) ? pitchBuf[i] : static_cast<TIdx>(sizeof(TElem));
                    viewPtrNative += offsetView[i - static_cast<TIdx>(1u)] * pitch;
                }
            }
            REQUIRE(reinterpret_cast<TElem*>(viewPtrNative) == alpaka::getPtrNative(view));
        }
    }

    template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
    auto testViewSubViewMutable(
        alpaka::ViewSubView<TDev, TElem, TDim, TIdx>& view,
        TBuf& buf,
        TDev const& dev,
        alpaka::Vec<TDim, TIdx> const& extentView,
        alpaka::Vec<TDim, TIdx> const& offsetView) -> void
    {
        testViewSubViewImmutable<TAcc>(view, buf, dev, extentView, offsetView);

        using Queue = alpaka::test::DefaultQueue<TDev>;
        Queue queue(dev);
        alpaka::test::testViewMutable<TAcc>(queue, view);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewNoOffset() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const extentBuf
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, extentBuf);

        auto const extentView = extentBuf;
        auto const offsetView = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(0));
        View view(buf);

        alpaka::test::testViewSubViewMutable<TAcc>(view, buf, dev, extentView, offsetView);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewOffset() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const extentBuf
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, extentBuf);

        auto const extentView
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentSubView>();
        auto const offsetView
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForOffset>();
        View view(buf, extentView, offsetView);

        alpaka::test::testViewSubViewMutable<TAcc>(view, buf, dev, extentView, offsetView);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewOffsetConst() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const extentBuf
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, extentBuf);

        auto const extentView
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentSubView>();
        auto const offsetView
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForOffset>();
        View const view(buf, extentView, offsetView);

        alpaka::test::testViewSubViewImmutable<TAcc>(view, buf, dev, extentView, offsetView);
    }
} // namespace alpaka::test
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

TEMPLATE_LIST_TEST_CASE("viewSubViewNoOffsetTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewSubViewNoOffset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewSubViewOffset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetConstTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewSubViewOffsetConst<TestType, float>();
}
