/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

TEST_CASE("move MdSpan", "[mem][mdspan][lifetime]")
{
    constexpr size_t size = 10;
    std::vector<int> data(size);
    int* ptr = data.data();
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    // mdspan needs to be trivial copyable, otherwise it cannot be passed to the kernel
    static_assert(std::is_trivially_copyable_v<MutMdSpan>);
    static_assert(std::is_trivially_copyable_v<ConstMdSpan>);

    MutMdSpan mdspan = MdSpan(ptr, extents, pitches);
    REQUIRE(mdspan);
    REQUIRE(mdspan.data() != nullptr);

    [[maybe_unused]] ConstMdSpan otherMdspan = std::move(mdspan);
    REQUIRE(mdspan);
    REQUIRE(mdspan.data() != nullptr);
    REQUIRE(otherMdspan);
    REQUIRE(otherMdspan.data() != nullptr);

    // because moving a mdspan does a copy, the original mdspan should be still valid after the move
    REQUIRE(mdspan.data() == otherMdspan.data());
    REQUIRE(mdspan.getExtents() == otherMdspan.getExtents());
    REQUIRE(mdspan.getPitches() == otherMdspan.getPitches());
}

TEST_CASE("move View", "[mem][view][lifetime]")
{
    constexpr size_t size = 10;
    std::vector<int> data(size);
    int* ptr = data.data();
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    MutView view(api::host, ptr, extents, pitches);
    REQUIRE(view);
    REQUIRE(view.data() != nullptr);

    [[maybe_unused]] ConstView constView = std::move(view);
    REQUIRE(view);
    REQUIRE(view.data() != nullptr);
    REQUIRE(constView);
    REQUIRE(constView.data() != nullptr);

    // because moving a view does a copy, the original view should be still valid after the move
    REQUIRE(view.data() == constView.data());
    REQUIRE(view.getExtents() == constView.getExtents());
    REQUIRE(view.getPitches() == constView.getPitches());
}

TEST_CASE("copy SharedBuffer", "[mem][sharedBuffer][lifetime]")
{
    onHost::SharedBuffer buffer = onHost::allocHost<int>(Vec<unsigned int, 1>{10});
    REQUIRE(buffer.getUseCount() == 1);

    onHost::SharedBuffer buffer2 = buffer;
    REQUIRE(buffer.getUseCount() == 2);
    REQUIRE(buffer);
    REQUIRE(buffer2.getUseCount() == 2);
    REQUIRE(buffer2);
}

TEST_CASE("move SharedBuffer", "[mem][sharedBuffer][lifetime]")
{
    onHost::SharedBuffer buffer = onHost::allocHost<int>(Vec<unsigned int, 1>{10});
    REQUIRE(buffer.getUseCount() == 1);

    onHost::SharedBuffer buffer2 = std::move(buffer);
    REQUIRE_FALSE(buffer);
    REQUIRE(buffer2.getUseCount() == 1);
    REQUIRE(buffer2);

    onHost::SharedBuffer buffer3(std::move(buffer2));
    REQUIRE_FALSE(buffer);
    REQUIRE_FALSE(buffer2);
    REQUIRE(buffer3.getUseCount() == 1);
    REQUIRE(buffer3);
}

struct LivingMemory
{
    // live counter:
    // 1 -> alive
    // 0 -> freed
    // negative -> double free
    int liveCounter = 1;
};

TEST_CASE("lifetime of shared memory", "[mem][sharedBuffer][lifetime]")
{
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(1);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    LivingMemory lvMem1;
    auto lvMemDeleter1 = [&lvMem1] { lvMem1.liveCounter -= 1; };
    {
        onHost::SharedBuffer buffer(api::host, &lvMem1, extents, pitches, lvMemDeleter1);
        REQUIRE(lvMem1.liveCounter == 1);
    }
    REQUIRE(lvMem1.liveCounter == 0);

    LivingMemory lvMem2;
    auto lvMemDeleter2 = [&lvMem2] { lvMem2.liveCounter -= 1; };
    onHost::SharedBuffer buffer2(api::host, &lvMem2, extents, pitches, lvMemDeleter2);
    {
        onHost::SharedBuffer buffer = buffer2;
        REQUIRE(buffer2.getUseCount() == 2);
        REQUIRE(lvMem2.liveCounter == 1);
    }
    REQUIRE(lvMem2.liveCounter == 1);
    REQUIRE(buffer2.getUseCount() == 1);

    LivingMemory lvMem3;
    auto lvMemDeleter3 = [&lvMem3] { lvMem3.liveCounter -= 1; };
    onHost::SharedBuffer buffer3(api::host, &lvMem3, extents, pitches, lvMemDeleter3);
    {
        onHost::SharedBuffer buffer(std::move(buffer3));
        REQUIRE(buffer.getUseCount() == 1);
        REQUIRE(lvMem3.liveCounter == 1);
    }
    REQUIRE(lvMem3.liveCounter == 0);
    REQUIRE_FALSE(buffer3);
}

void funcCopyByValue(auto buffer, long const expectedUseCount)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expectedUseCount);
}

void funcReference(auto& buffer, long const expectedUseCount)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expectedUseCount);
}

void funcConstReference(auto const& buffer, long const expectedUseCount)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expectedUseCount);
}

/** Takes the buffer as universal reference but do not assign to a lvalue.
 * Therefore, the buffer is still valid after the function call.
 *
 * see: https://alagner.github.io/2024/03/14/Passing-arguments-by-move.html
 */
void funcUniversalRefBorrow(auto&& buffer, long const expectedUseCount)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expectedUseCount);
}

/** Takes the buffer as universal reference and assign it to a lvalue.
 * Therefore, the buffer is not valid after the function call anymore.
 *
 */
void funcUniversalRefMoved(auto&& buffer, long const expectedUseCount)
{
    REQUIRE(buffer);
    auto tmpBuffer = std::move(buffer);
    REQUIRE_FALSE(buffer);
    REQUIRE(tmpBuffer);
    REQUIRE(tmpBuffer.getUseCount() == expectedUseCount);
}

TEST_CASE("pass shared memory to function", "[mem][sharedBuffer][lifetime]")
{
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(1);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    LivingMemory lvMem1;
    auto lvMemDeleter1 = [&lvMem1] { lvMem1.liveCounter -= 1; };
    onHost::SharedBuffer buffer1(api::host, &lvMem1, extents, pitches, lvMemDeleter1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcCopyByValue(buffer1, 2);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lvMem1.liveCounter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcReference(buffer1, 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lvMem1.liveCounter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcConstReference(buffer1, 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lvMem1.liveCounter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcUniversalRefBorrow(buffer1, 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lvMem1.liveCounter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcUniversalRefBorrow(std::move(buffer1), 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lvMem1.liveCounter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcUniversalRefMoved(std::move(buffer1), 1);
    REQUIRE_FALSE(buffer1);
    REQUIRE(lvMem1.liveCounter == 0);

    LivingMemory lvMem2;
    auto lvMemDeleter2 = [&lvMem2] { lvMem2.liveCounter -= 1; };
    funcUniversalRefMoved(onHost::SharedBuffer{api::host, &lvMem2, extents, pitches, lvMemDeleter2}, 1);
    REQUIRE(lvMem2.liveCounter == 0);
}
