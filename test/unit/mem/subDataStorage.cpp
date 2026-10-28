/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/testMacros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <type_traits>

TEST_CASE("1D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int x = 10;
    auto buffer0 = alpaka::onHost::allocHost<int>(x);
    for(auto i = 0; i < x; ++i)
    {
        buffer0[i] = i;
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, single integer, define extent")
    {
        constexpr int subSize = 8;
        auto subView0 = view0.getSubView(subSize);
        REQUIRE(subView0.getExtents() == alpaka::Vec{subSize});
        for(auto i = 0; i < subSize; ++i)
        {
            REQUIRE_MESSAGE(subView0[i] == i, "i=" << i);
        }
    }

    SECTION("getSubView, single integer, define offset and extent")
    {
        constexpr int offset = 2;
        constexpr int end = 7;
        auto subView0 = view0.getSubView(offset, end);

        REQUIRE(subView0.getExtents() == alpaka::Vec{end});
        for(auto i = 0; i < subView0.getExtents()[0]; ++i)
        {
            REQUIRE_MESSAGE(subView0[i] == i + offset, "i=" << i);
        }
    }

    SECTION("getSubView, zero extent in 1D stays empty and does not iterate")
    {
        // Zero-sized 1D subviews should preserve the requested extent and produce an empty host iteration range.
        auto subView0 = view0.getSubView(0);

        REQUIRE(subView0.getExtents() == alpaka::Vec{0});

        auto count = 0;
        for([[maybe_unused]] auto&& value : subView0)
            ++count;

        REQUIRE(count == 0);
    }
}

TEST_CASE("3D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int z = 3;
    constexpr int y = 5;
    constexpr int x = 4;
    alpaka::Vec totalExtents{z, y, x};
    REQUIRE(totalExtents.z() == z);
    REQUIRE(totalExtents.y() == y);
    REQUIRE(totalExtents.x() == x);

    auto buffer0 = alpaka::onHost::allocHost<int>(totalExtents);

    for(auto vec : alpaka::IdxRange{totalExtents})
    {
        buffer0[vec] = vec.z() * 100 + vec.y() * 10 + vec.x();
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, define extent")
    {
        alpaka::Vec extentsSubview0{z - 1, y - 1, x - 1};
        auto subView0 = view0.getSubView(extentsSubview0);
        REQUIRE(subView0.getApi() == view0.getApi());

        STATIC_REQUIRE(subView0.dim() == extentsSubview0.dim());
        REQUIRE(subView0.getExtents() == extentsSubview0);

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            REQUIRE_MESSAGE(subView0[vec] == vec.z() * 100 + vec.y() * 10 + vec.x(), "vec=" << vec);
        }
    }

    SECTION("getSubView, define offset and extent")
    {
        alpaka::Vec offsetSubview0{1, 2, 1};
        alpaka::Vec extentsSubview0{z - 1, y - 3, x - 1};

        auto subView0 = view0.getSubView(offsetSubview0, extentsSubview0);
        REQUIRE(subView0.getApi() == view0.getApi());

        STATIC_REQUIRE(subView0.dim() == extentsSubview0.dim());
        REQUIRE(subView0.getExtents() == extentsSubview0);

        REQUIRE(subView0[alpaka::Vec{0, 0, 0}] == view0[offsetSubview0]);
        REQUIRE(
            subView0[alpaka::Vec{0, 1, 0}]
            == view0[alpaka::Vec{offsetSubview0.z(), offsetSubview0.y() + 1, offsetSubview0.x()}]);
        REQUIRE(
            subView0[extentsSubview0 - alpaka::Vec{1, 1, 1}]
            == view0[offsetSubview0 + extentsSubview0 - alpaka::Vec{1, 1, 1}]);

        REQUIRE(offsetSubview0.x() + extentsSubview0.x() <= view0.getExtents().x());
        REQUIRE(offsetSubview0.y() + extentsSubview0.y() <= view0.getExtents().y());
        REQUIRE(offsetSubview0.z() + extentsSubview0.z() <= view0.getExtents().z());

        auto counter = offsetSubview0;
        for(auto vec : alpaka::IdxRange{subView0.getExtents()})
        {
            REQUIRE_MESSAGE(subView0[vec] == counter.z() * 100 + counter.y() * 10 + counter.x(), "vec=" << vec);
            counter.x()++;
            if(counter.x() == offsetSubview0.x() + extentsSubview0.x())
            {
                counter.y()++;
                counter.x() = offsetSubview0.x();
                if(counter.y() == offsetSubview0.y() + extentsSubview0.y())
                {
                    counter.z()++;
                    counter.y() = offsetSubview0.y();
                }
            }
        }
    }

    SECTION("getSubView, zero extent in 3D stays empty and does not iterate")
    {
        // Zero in any dimension should still create a valid empty subview for host iteration.
        alpaka::Vec extentsSubview0{z, 0, x};
        auto subView0 = view0.getSubView(extentsSubview0);

        REQUIRE(subView0.getExtents() == extentsSubview0);

        auto count = 0;
        for([[maybe_unused]] auto&& value : subView0)
            ++count;

        REQUIRE(count == 0);
    }

    SECTION("getSubView with offset writes through to the parent storage")
    {
        // Offset subviews must alias the parent data so writes land at shifted coordinates in the original view.
        alpaka::Vec offsetSubview0{1, 2, 1};
        alpaka::Vec extentsSubview0{1, 2, 2};
        auto subView0 = view0.getSubView(offsetSubview0, extentsSubview0);

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            subView0[vec] = 900 + static_cast<int>(alpaka::linearize(extentsSubview0, vec));
        }

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            auto const parentIdx = offsetSubview0 + vec;
            REQUIRE(view0[parentIdx] == 900 + static_cast<int>(alpaka::linearize(extentsSubview0, vec)));
        }
    }

    SECTION("const getSubView with offset stays read-only and reads shifted values")
    {
        // `getSubView() const` should propagate constness to the returned view while still reading the shifted region.
        auto const& constView0 = view0;
        alpaka::Vec offsetSubview0{1, 2, 1};
        alpaka::Vec extentsSubview0{2, 2, 3};
        auto subView0 = constView0.getSubView(offsetSubview0, extentsSubview0);

        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[alpaka::Vec{0, 0, 0}])>>);

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            auto const parentIdx = offsetSubview0 + vec;
            REQUIRE(subView0[vec] == view0[parentIdx]);
        }
    }
}

TEST_CASE("alpaka::View::getSubView keeps pitches, pointer, and alignment contracts", "[mem][view][SubDataStorage]")
{
    alignas(32) std::array<int, std::size_t{2} * 3 * 4> storage{};
    for(std::size_t i = 0; i < storage.size(); ++i)
    {
        storage[i] = static_cast<int>(i);
    }

    auto view0 = alpaka::makeView(alpaka::api::host, storage.data(), alpaka::Vec{2, 3, 4}, alpaka::Alignment<32>{});

    SECTION("offset subviews preserve pitches and drop to plain alignment")
    {
        // A shifted origin can break stronger alignment guarantees, but the pitch layout must still match the parent.
        auto const offsetSubview0 = view0.getSubView(alpaka::Vec{1, 1, 1}, alpaka::Vec{1, 2, 3});

        REQUIRE(offsetSubview0.getPitches() == view0.getPitches());
        REQUIRE(offsetSubview0.data() == &view0[alpaka::Vec{1, 1, 1}]);

        STATIC_REQUIRE(offsetSubview0.getAlignment().get<int>() == alpaka::Alignment<>::get<int>());
    }

    SECTION("extent-only subviews keep the original pointer and alignment")
    {
        // Cropping only by extent must not move the pointer or weaken the parent alignment contract.
        auto const subView0 = view0.getSubView(alpaka::Vec{2, 2, 3});

        REQUIRE(subView0.data() == view0.data());
        REQUIRE(subView0.getPitches() == view0.getPitches());

        STATIC_REQUIRE(subView0.getAlignment().get<int>() == alpaka::Alignment<32>::get<int>());
    }
}
