/* Copyright 2026 René Widera, Felice Pantaleo
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace
{
    using Single = alpaka::rand::engine::Philox4x32x10;
    using Vector = alpaka::rand::engine::Philox4x32x10Vector;
    using Params = alpaka::rand::engine::internal::PhiloxParams<4, 32, 10>;
    using Stateless = alpaka::rand::engine::internal::PhiloxStateless<Params>;
    using Counter = Stateless::Counter;
    using Key = Stateless::Key;

    constexpr std::size_t numberOfFixedNumbers = 8;

    /** Fixed numbers of the stateful single-number engine.
     *
     * Number n is word n % 4 of the block with the 128-bit counter (subsequence << 64) + 1 + n / 4. The values
     * were computed with an independent Philox4x32-10 implementation which reproduces the Random123 known
     * answers.
     */
    struct FixedNumbers
    {
        std::uint64_t seed;
        std::uint64_t subsequence;
        std::uint64_t offset;
        std::uint32_t numbers[numberOfFixedNumbers];
    };

    constexpr FixedNumbers fixedNumbers[] = {
        {0x0123'4567'89ab'cdefull,
         (std::uint64_t{1} << 32) + 7,
         (std::uint64_t{1} << 34) + 5,
         {0xf058'c646, 0xc889'd089, 0x75f2'ac85, 0xf9fc'55b6, 0x08b8'bb9c, 0xd605'e12c, 0x4b05'e171, 0x9a80'40db}},
        {0xfedc'ba98'7654'3210ull,
         0xffff'ffff'ffff'ffffull,
         0xffff'ffff'ffff'ffffull,
         {0x3f50'06e0, 0x35a1'b7d8, 0xb526'0185, 0x5bc5'1535, 0x59b6'4fb2, 0x35d7'33e6, 0x3515'ae08, 0xb650'f3a5}},
    };

    auto draw(Single& engine, std::size_t count) -> std::vector<std::uint32_t>
    {
        std::vector<std::uint32_t> numbers;
        for(std::size_t i = 0; i < count; ++i)
            numbers.push_back(engine());
        return numbers;
    }
} // namespace

// The engine is constexpr, so the Random123 known answer can be checked at compile time.
static_assert(Stateless::generate(Counter{0, 0, 0, 0}, Key{0, 0})[0] == 0x6627'e8d5);
static_assert(Stateless::generate(Counter{0, 0, 0, 0}, Key{0, 0})[3] == 0x9b00'dbd8);

TEST_CASE("PhiloxRandom123KnownAnswer", "[rand]")
{
    // Random123 tests/kat_vectors, philox4x32-10, counter 0 and key 0.
    auto const result = Stateless::generate(Counter{0, 0, 0, 0}, Key{0, 0});
    CHECK(result[0] == 0x6627'e8d5);
    CHECK(result[1] == 0xe169'c58d);
    CHECK(result[2] == 0xbc57'ac4c);
    CHECK(result[3] == 0x9b00'dbd8);
}

TEST_CASE("PhiloxStatefulFixedNumbers", "[rand]")
{
    for(auto const& fixed : fixedNumbers)
    {
        Single engine(fixed.seed, fixed.subsequence, fixed.offset);
        for(std::size_t i = 0; i < numberOfFixedNumbers; ++i)
            CHECK(engine() == fixed.numbers[i]);
    }
}

TEST_CASE("PhiloxOffsetEqualsDiscard", "[rand]")
{
    constexpr std::uint64_t seed = 42;
    for(std::uint64_t const subsequence : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}})
    {
        Single reference(seed, subsequence, 0);
        auto const expected = draw(reference, 96);
        for(std::uint64_t offset = 0; offset <= 24; ++offset)
        {
            Single engine(seed, subsequence, offset);
            auto const actual = draw(engine, 8);
            for(std::size_t i = 0; i < actual.size(); ++i)
                CHECK(actual[i] == expected[offset + i]);
        }
    }
}

TEST_CASE("PhiloxSubsequencesDoNotOverlap", "[rand]")
{
    // Subsequence 5 + 2^32 at offset 20 must not continue subsequence 5 after 20 numbers.
    Single low(1, 5, 0);
    auto const lowNumbers = draw(low, 28);
    Single high(1, (std::uint64_t{1} << 32) + 5, 20);
    auto const highNumbers = draw(high, 8);
    CHECK(std::vector<std::uint32_t>(lowNumbers.begin() + 20, lowNumbers.end()) != highNumbers);
}

TEST_CASE("PhiloxVectorMatchesSingle", "[rand]")
{
    constexpr std::uint64_t seed = 0x0123'4567'89ab'cdefull;
    constexpr std::uint64_t subsequence = (std::uint64_t{1} << 32) + 7;
    for(std::uint64_t offset = 0; offset < 3; ++offset)
    {
        Vector vector(seed, subsequence, offset);
        for(unsigned v = 0; v < 4; ++v)
        {
            auto const result = vector();
            Single single(seed, subsequence, 4 * (offset + v));
            for(unsigned i = 0; i < 4; ++i)
                CHECK(result[i] == single());
        }
    }
}

TEST_CASE("PhiloxVectorTakesTheFullSeed", "[rand]")
{
    constexpr std::uint64_t highBit = std::uint64_t{1} << 32;
    CHECK(Vector(43)() != Vector(highBit + 43)());
    CHECK(Vector(43, 7)() != Vector(43, highBit + 7)());
    CHECK(Vector(43, 7, 0)() != Vector(43, 7, highBit)());
}

TEST_CASE("PhiloxRandom123KnownAnswersWithAllWords", "[rand]")
{
    // Random123 tests/kat_vectors, philox4x32-10: all words set, and the digits of pi.
    struct KnownAnswer
    {
        Counter counter;
        Key key;
        Counter expected;
    };

    constexpr KnownAnswer knownAnswers[] = {
        {{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
         {0xffff'ffff, 0xffff'ffff},
         {0x408f'276d, 0x41c8'3b0e, 0xa20b'c7c6, 0x6d54'51fd}},
        {{0x243f'6a88, 0x85a3'08d3, 0x1319'8a2e, 0x0370'7344},
         {0xa409'3822, 0x299f'31d0},
         {0xd16c'fe09, 0x94fd'cceb, 0x5001'e420, 0x2412'6ea1}},
    };
    for(auto const& answer : knownAnswers)
        CHECK(Stateless::generate(answer.counter, answer.key) == answer.expected);
}

namespace
{
    // Gives access to the protected skip functions of the single-number engine.
    class SingleProbe : public alpaka::rand::engine::internal::PhiloxSingle<Params>
    {
    public:
        using Base = alpaka::rand::engine::internal::PhiloxSingle<Params>;
        using Base::Base;

        constexpr void skipBlocks(std::uint64_t const blocks)
        {
            this->skip4(blocks);
        }

        constexpr void skipNumbers(std::uint64_t const numbers)
        {
            this->skip(numbers);
        }
    };
} // namespace

TEST_CASE("PhiloxSkip4CarriesIntoTheHighWords", "[rand]")
{
    SingleProbe engine;
    engine.state.counter = Counter{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 5};
    engine.skipBlocks(1);
    CHECK(engine.state.counter == Counter{0, 0, 0, 6});
    engine.state.counter = Counter{0xffff'fffe, 0xffff'ffff, 7, 9};
    engine.skipBlocks(3);
    CHECK(engine.state.counter == Counter{1, 0, 8, 9});
    engine.state.counter = Counter{5, 0, 0, 0};
    engine.skipBlocks(0xffff'ffff'ffff'ffff);
    CHECK(engine.state.counter == Counter{4, 0, 1, 0});
    engine.state.counter = Counter{5, 0, 0xffff'ffff, 0};
    engine.skipBlocks(0);
    CHECK(engine.state.counter == Counter{5, 0, 0xffff'ffff, 0});
}

TEST_CASE("PhiloxSkipEqualsDiscard", "[rand]")
{
    constexpr std::uint64_t seed = 42;
    constexpr std::uint64_t subsequence = 5;
    Single reference(seed, subsequence, 0);
    auto const expected = draw(reference, 32);
    // Each start position in the block, and skips that stay in the block, end it and go past it.
    for(std::size_t draws = 0; draws < 8; ++draws)
    {
        for(std::uint64_t skip = 0; skip <= 17; ++skip)
        {
            SingleProbe engine(seed, subsequence, 0);
            for(std::size_t i = 0; i < draws; ++i)
                static_cast<void>(engine());
            engine.skipNumbers(skip);
            INFO("draws " << draws << " skip " << skip);
            for(std::size_t i = 0; i < 4; ++i)
                CHECK(engine() == expected[draws + skip + i]);
        }
    }
    // Numbers 2^64 and 2^64 + 1 of seed 3, subsequence 9 are words 0 and 1 of block 2^62 + 1.
    SingleProbe engine(3, 9, 0);
    static_cast<void>(engine());
    engine.skipNumbers(0xffff'ffff'ffff'ffff);
    CHECK(engine() == 0x965b'c3d3);
    CHECK(engine() == 0x5b33'dcdf);
}
