/* Copyright 2024 Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/rand/RandDefault.hpp>
#include <alpaka/rand/RandPhilox.hpp>
#include <alpaka/rand/RandPhiloxStateless.hpp>
#include <alpaka/rand/RandStdLib.hpp>
#include <alpaka/rand/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

class PhiloxTest
{
protected:
    alpaka::rand::Philox4x32x10 statefulSingleEngine;
    alpaka::rand::Philox4x32x10Vector statefulVectorEngine;
};

TEST_CASE_METHOD(PhiloxTest, "HostStatefulVectorEngineTest")
{
    auto const resultVec = statefulVectorEngine();
    for(auto& result : resultVec)
    {
        REQUIRE(result >= statefulVectorEngine.min());
        REQUIRE(result <= statefulVectorEngine.max());
    }
}

TEST_CASE_METHOD(PhiloxTest, "HostStatefulSingleEngineTest")
{
    auto const result = statefulSingleEngine();
    REQUIRE(result >= statefulSingleEngine.min());
    REQUIRE(result <= statefulSingleEngine.max());
}

TEST_CASE("HostStatelessEngineTest")
{
    using Gen = alpaka::rand::PhiloxStateless4x32x10Vector;
    using Key = typename Gen::Key;
    using Counter = typename Gen::Counter;
    Key key = {42, 12345};
    Counter counter1 = {6789, 321, 0, 0};
    auto const result1 = Gen::generate(counter1, key);
    Counter counter2 = {6789, 321, 0, 1};
    auto const result2 = Gen::generate(counter2, key);
    // Make sure that the inputs are really expected to lead to different results.
    REQUIRE(result1 != result2);
}

template<typename T>
class PhiloxTestKernelSingle
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            static_cast<void>(acc);
            alpaka::rand::UniformReal<T> dist;
            auto const result = dist(gen);
            ALPAKA_CHECK(*success, static_cast<T>(0.0) <= result);
            ALPAKA_CHECK(*success, static_cast<T>(1.0) > result);
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Philox generator for accelerator
        auto generator = alpaka::rand::Philox4x32x10(42, 12345, 6789);
        genNumbers<TAcc, decltype(generator)>(acc, success, generator);
    }
};

template<typename T>
class PhiloxTestKernelVector
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            static_cast<void>(acc);
            using DistributionResult = typename T_Generator::template ResultContainer<T>;
            alpaka::rand::UniformReal<DistributionResult> dist;
            auto const result = dist(gen);
            for(auto& element : result)
            {
                ALPAKA_CHECK(*success, static_cast<T>(0.0) <= element);
                ALPAKA_CHECK(*success, static_cast<T>(1.0) > element);
            }
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Philox generator for accelerator
        auto generator = alpaka::rand::Philox4x32x10Vector(42, 12345, 6789);
        genNumbers<TAcc, decltype(generator)>(acc, success, generator);
    }
};

class PhiloxTestKernelStateless
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success) const
    {
        {
            static_cast<void>(acc);

            using Gen = alpaka::rand::PhiloxStateless4x32x10Vector;
            using Key = typename Gen::Key;
            using Counter = typename Gen::Counter;

            Key key = {42, 12345};
            Counter counter = {6789, 321, 0, 0};
            auto const result = Gen::generate(counter, key);

            size_t check = 0;
            for(auto& element : result)
            {
                check += element;
            }
            // Make sure the sequence is not in fact supposed to generate {0,0,0,0}.
            ALPAKA_CHECK(*success, check != 0);
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        genNumbers<TAcc>(acc, success);
    }
};

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorStatelessIsWorking", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelStateless kernel;

    REQUIRE(fixture(kernel));
}

using TestScalars = std::tuple<float, double>;
using TestTypes = alpaka::meta::CartesianProduct<std::tuple, alpaka::test::TestAccs, TestScalars>;

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorSingleIsWorking", "[rand]", TestTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using DataType = std::tuple_element_t<1, TestType>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelSingle<DataType> kernel;

    REQUIRE(fixture(kernel));
}

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorVectorIsWorking", "[rand]", TestTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using DataType = std::tuple_element_t<1, TestType>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelVector<DataType> kernel;

    REQUIRE(fixture(kernel));
}

namespace
{
    // Known-answer vectors for philox4x32-10 from the Random123 distribution, file tests/kat_vectors.
    struct PhiloxKnownAnswer
    {
        std::uint32_t counter[4];
        std::uint32_t key[2];
        std::uint32_t expected[4];
    };

    constexpr std::size_t numPhiloxKnownAnswers = 3;

    ALPAKA_FN_HOST_ACC constexpr auto philoxKnownAnswer(std::size_t index) -> PhiloxKnownAnswer
    {
        switch(index)
        {
        case 0:
            return {
                {0x0000'0000, 0x0000'0000, 0x0000'0000, 0x0000'0000},
                {0x0000'0000, 0x0000'0000},
                {0x6627'e8d5, 0xe169'c58d, 0xbc57'ac4c, 0x9b00'dbd8}};
        case 1:
            return {
                {0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 0xffff'ffff},
                {0xffff'ffff, 0xffff'ffff},
                {0x408f'276d, 0x41c8'3b0e, 0xa20b'c7c6, 0x6d54'51fd}};
        default:
            return {
                {0x243f'6a88, 0x85a3'08d3, 0x1319'8a2e, 0x0370'7344},
                {0xa409'3822, 0x299f'31d0},
                {0xd16c'fe09, 0x94fd'cceb, 0x5001'e420, 0x2412'6ea1}};
        }
    }

    template<typename TEngine>
    auto drawNumbers(TEngine engine, std::size_t count) -> std::vector<std::uint32_t>
    {
        std::vector<std::uint32_t> numbers(count);
        for(auto& number : numbers)
        {
            number = engine();
        }
        return numbers;
    }

    // An engine that always returns its largest value.
    struct MaxValueEngine
    {
        using result_type = std::uint32_t;

        ALPAKA_FN_HOST_ACC auto operator()() -> std::uint32_t
        {
            return 0xffff'ffff;
        }

        ALPAKA_FN_HOST_ACC static constexpr auto min() -> std::uint32_t
        {
            return 0;
        }

        ALPAKA_FN_HOST_ACC static constexpr auto max() -> std::uint32_t
        {
            return 0xffff'ffff;
        }
    };
} // namespace

TEST_CASE("HostStatelessEngineKnownAnswers", "[rand]")
{
    using Gen = alpaka::rand::PhiloxStateless4x32x10Vector;
    for(std::size_t index = 0; index < numPhiloxKnownAnswers; ++index)
    {
        auto const answer = philoxKnownAnswer(index);
        auto const result = Gen::generate(
            typename Gen::Counter{answer.counter[0], answer.counter[1], answer.counter[2], answer.counter[3]},
            typename Gen::Key{answer.key[0], answer.key[1]});
        for(std::size_t i = 0; i < 4; ++i)
        {
            INFO("vector " << index << " word " << i);
            CHECK(result[i] == answer.expected[i]);
        }
    }
}

class PhiloxTestKernelKnownAnswers
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        using Gen = alpaka::rand::PhiloxStateless4x32x10Vector;
        for(std::size_t index = 0; index < numPhiloxKnownAnswers; ++index)
        {
            auto const answer = philoxKnownAnswer(index);
            auto const result = Gen::generate(
                typename Gen::Counter{answer.counter[0], answer.counter[1], answer.counter[2], answer.counter[3]},
                typename Gen::Key{answer.key[0], answer.key[1]});
            for(std::size_t i = 0; i < 4; ++i)
            {
                ALPAKA_CHECK(*success, result[i] == answer.expected[i]);
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorKnownAnswers", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelKnownAnswers kernel;

    REQUIRE(fixture(kernel));
}

TEST_CASE("HostStatefulSingleEngineOffsetEqualsDiscard", "[rand]")
{
    constexpr std::uint64_t seed = 42;
    for(std::uint64_t const subsequence :
        {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, (std::uint64_t{1} << 32) + 5})
    {
        auto const reference = drawNumbers(alpaka::rand::Philox4x32x10(seed, subsequence, 0), 80);
        for(std::uint64_t offset = 0; offset <= 64; ++offset)
        {
            auto const numbers = drawNumbers(alpaka::rand::Philox4x32x10(seed, subsequence, offset), 16);
            for(std::size_t i = 0; i < numbers.size(); ++i)
            {
                INFO("subsequence " << subsequence << " offset " << offset << " number " << i);
                CHECK(numbers[i] == reference[offset + i]);
            }
        }
    }
}

TEST_CASE("HostStatefulSingleEngineSubsequencesDoNotOverlap", "[rand]")
{
    // Subsequence 5 + 2^32 at offset 20 must not repeat subsequence 5 after 20 numbers.
    auto const first = drawNumbers(alpaka::rand::Philox4x32x10(1, 5, 0), 28);
    auto const second = drawNumbers(alpaka::rand::Philox4x32x10(1, (std::uint64_t{1} << 32) + 5, 20), 8);
    CHECK(std::vector<std::uint32_t>(first.begin() + 20, first.end()) != second);
}

TEST_CASE("HostStatefulVectorEngineUsesTheFullSeed", "[rand]")
{
    constexpr std::uint64_t highBit = std::uint64_t{1} << 32;
    auto const low = alpaka::rand::Philox4x32x10Vector(43, 0, 0)();
    auto const high = alpaka::rand::Philox4x32x10Vector(highBit + 43, 0, 0)();
    CHECK(low != high);
    auto const lowSubsequence = alpaka::rand::Philox4x32x10Vector(43, 7, 0)();
    auto const highSubsequence = alpaka::rand::Philox4x32x10Vector(43, highBit + 7, 0)();
    CHECK(lowSubsequence != highSubsequence);
}

TEST_CASE("UniformRealExcludesOne", "[rand]")
{
    MaxValueEngine engine;
    CHECK(alpaka::rand::UniformReal<float>()(engine) < 1.0f);
    CHECK(alpaka::rand::UniformReal<double>()(engine) < 1.0);
}

namespace
{
    // Expected numbers of the stateful engines. Number n of the single-number engine is word n % 4 of
    // generate({b low, b high, s low, s high}, {seed low, seed high}) with block b = 1 + n / 4 and subsequence s.
    // Vector i of the vector engine at offset n is the block of the 128-bit counter (s << 64) + 1 + n + i.
    struct PhiloxStatefulKnownNumbers
    {
        std::uint64_t seed;
        std::uint64_t subsequence;
        std::uint64_t offset;
        std::uint32_t single[8];
        std::uint32_t vector[8];
    };

    constexpr std::size_t numPhiloxStatefulKnownNumbers = 3;

    ALPAKA_FN_HOST_ACC constexpr auto philoxStatefulKnownNumbers(std::size_t index) -> PhiloxStatefulKnownNumbers
    {
        switch(index)
        {
        case 0:
            return {
                0x0123'4567'89ab'cdef,
                (std::uint64_t{1} << 32) + 7,
                (std::uint64_t{1} << 34) + 5,
                {0xf058'c646,
                 0xc889'd089,
                 0x75f2'ac85,
                 0xf9fc'55b6,
                 0x08b8'bb9c,
                 0xd605'e12c,
                 0x4b05'e171,
                 0x9a80'40db},
                {0x0ea1'2801,
                 0x96de'f29d,
                 0x3365'410b,
                 0xee0b'7a5d,
                 0x678c'68b6,
                 0x8d57'b9cd,
                 0xae5d'ec9b,
                 0xa7c7'3973}};
        case 1:
            return {
                0xfedc'ba98'7654'3210,
                0xffff'ffff'ffff'ffff,
                0xffff'ffff'ffff'ffff,
                {0x3f50'06e0,
                 0x35a1'b7d8,
                 0xb526'0185,
                 0x5bc5'1535,
                 0x59b6'4fb2,
                 0x35d7'33e6,
                 0x3515'ae08,
                 0xb650'f3a5},
                {0xba4b'834c,
                 0x0559'bf7f,
                 0xf35d'3e34,
                 0x349c'2ad4,
                 0x29fc'c21d,
                 0x0611'21ec,
                 0x300a'371d,
                 0xcf79'8377}};
        default:
            return {
                42,
                0,
                0,
                {0xfcdb'2127,
                 0x53ba'6cfd,
                 0x838f'5a6e,
                 0x744e'06fb,
                 0xd36c'0225,
                 0xa887'5dcb,
                 0x9a4d'6d99,
                 0xc609'a559},
                {0xfcdb'2127,
                 0x53ba'6cfd,
                 0x838f'5a6e,
                 0x744e'06fb,
                 0xd36c'0225,
                 0xa887'5dcb,
                 0x9a4d'6d99,
                 0xc609'a559}};
        }
    }

    // Gives access to the protected skip functions of the single-number engine.
    class PhiloxSingleProbe : public alpaka::rand::engine::PhiloxSingle<alpaka::rand::engine::PhiloxParams<4, 32, 10>>
    {
    public:
        using Base = alpaka::rand::engine::PhiloxSingle<alpaka::rand::engine::PhiloxParams<4, 32, 10>>;
        using Base::Base;

        void skipBlocks(std::uint64_t const blocks)
        {
            this->skip4(blocks);
        }

        void skipNumbers(std::uint64_t const numbers)
        {
            this->skip(numbers);
        }
    };

    // An engine that always returns one value, with the range [TMin, TMax].
    template<typename TValue, TValue TMin, TValue TMax>
    struct FixedValueEngine
    {
        using result_type = TValue;

        TValue value;

        ALPAKA_FN_HOST_ACC auto operator()() -> TValue
        {
            return value;
        }

        ALPAKA_FN_HOST_ACC static constexpr auto min() -> TValue
        {
            return TMin;
        }

        ALPAKA_FN_HOST_ACC static constexpr auto max() -> TValue
        {
            return TMax;
        }
    };
} // namespace

TEST_CASE("HostStatefulEnginesKnownNumbers", "[rand]")
{
    for(std::size_t index = 0; index < numPhiloxStatefulKnownNumbers; ++index)
    {
        auto const known = philoxStatefulKnownNumbers(index);
        alpaka::rand::Philox4x32x10 single(known.seed, known.subsequence, known.offset);
        for(std::size_t i = 0; i < 8; ++i)
        {
            INFO("case " << index << " number " << i);
            CHECK(single() == known.single[i]);
        }
        alpaka::rand::Philox4x32x10Vector vector(known.seed, known.subsequence, known.offset);
        for(std::size_t v = 0; v < 2; ++v)
        {
            auto const result = vector();
            for(std::size_t i = 0; i < 4; ++i)
            {
                INFO("case " << index << " vector " << v << " word " << i);
                CHECK(result[i] == known.vector[4 * v + i]);
            }
        }
    }
}

class PhiloxTestKernelStatefulKnownNumbers
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        for(std::size_t index = 0; index < numPhiloxStatefulKnownNumbers; ++index)
        {
            auto const known = philoxStatefulKnownNumbers(index);
            alpaka::rand::Philox4x32x10 single(known.seed, known.subsequence, known.offset);
            for(std::size_t i = 0; i < 8; ++i)
            {
                ALPAKA_CHECK(*success, single() == known.single[i]);
            }
            alpaka::rand::Philox4x32x10Vector vector(known.seed, known.subsequence, known.offset);
            for(std::size_t v = 0; v < 2; ++v)
            {
                auto const result = vector();
                for(std::size_t i = 0; i < 4; ++i)
                {
                    ALPAKA_CHECK(*success, result[i] == known.vector[4 * v + i]);
                }
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorStatefulKnownNumbers", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelStatefulKnownNumbers kernel;

    REQUIRE(fixture(kernel));
}

TEST_CASE("HostPhiloxSkip4CarriesIntoTheHighWords", "[rand]")
{
    using Counter = PhiloxSingleProbe::Counter;
    PhiloxSingleProbe engine;
    engine.state.counter = Counter{0xffff'ffff, 0xffff'ffff, 0xffff'ffff, 5};
    engine.skipBlocks(1);
    CHECK(engine.state.counter == Counter{0, 0, 0, 6});
    engine.state.counter = Counter{0xffff'fffe, 0xffff'ffff, 7, 9};
    engine.skipBlocks(3);
    CHECK(engine.state.counter == Counter{1, 0, 8, 9});
    engine.state.counter = Counter{5, 0, 0, 0};
    engine.skipBlocks(0xffff'ffff'ffff'ffff);
    CHECK(engine.state.counter == Counter{4, 0, 1, 0});
}

TEST_CASE("HostPhiloxSkipToNumberTwoToTheSixtyFour", "[rand]")
{
    // Seed 3, subsequence 9: numbers 2^64 and 2^64 + 1 are words 0 and 1 of block 2^62 + 1.
    PhiloxSingleProbe engine(3, 9, 0);
    static_cast<void>(engine());
    engine.skipNumbers(0xffff'ffff'ffff'ffff);
    CHECK(engine() == 0x965b'c3d3);
    CHECK(engine() == 0x5b33'dcdf);
}

TEST_CASE("UniformUintNarrowTypesAndBool", "[rand]")
{
    alpaka::rand::Philox4x32x10 engine(12345, 6789);
    alpaka::rand::distribution::gpu::UniformUint<std::uint16_t> narrow;
    std::uint16_t bits = 0;
    for(int i = 0; i < 64; ++i)
    {
        bits = static_cast<std::uint16_t>(bits | narrow(engine));
    }
    CHECK(bits == 0xffff);
    alpaka::rand::distribution::gpu::UniformUint<bool> oneBit;
    int count = 0;
    for(int i = 0; i < 64; ++i)
    {
        count += oneBit(engine) ? 1 : 0;
    }
    CHECK(count > 16);
    CHECK(count < 48);
}

TEST_CASE("FoldTo32BitsKeepsSmallValuesAndSeparatesHalves", "[rand]")
{
    using alpaka::rand::engine::trait::foldTo32Bits;
    CHECK(foldTo32Bits(0) == 0);
    CHECK(foldTo32Bits(12345) == 12345);
    CHECK(foldTo32Bits(0xffff'ffff) == 0xffff'ffff);
    constexpr std::uint64_t seven = 7;
    constexpr std::uint64_t nine = 9;
    CHECK(foldTo32Bits((seven << 32) | nine) != foldTo32Bits((nine << 32) | seven));
    CHECK(foldTo32Bits((std::uint64_t{42} << 32) | 42) != foldTo32Bits(0));
    CHECK(foldTo32Bits(std::uint64_t{1} << 32) != foldTo32Bits(1));
    CHECK(foldTo32Bits(0xffff'ffff'ffff'ffff) != foldTo32Bits(0));
    auto first = alpaka::rand::engine::createDefault(alpaka::rand::MersenneTwister{}, (seven << 32) | nine);
    auto second = alpaka::rand::engine::createDefault(alpaka::rand::MersenneTwister{}, (nine << 32) | seven);
    CHECK(first() != second());
}

TEST_CASE("UniformRealStaysBelowTheUpperLimit", "[rand]")
{
    MaxValueEngine engine;
    float const unit = alpaka::rand::UniformReal<float>(1.0f, 2.0f)(engine);
    CHECK(unit >= 1.0f);
    CHECK(unit < 2.0f);
    CHECK(alpaka::rand::UniformReal<float>(100.0f, 101.0f)(engine) < 101.0f);
    CHECK(alpaka::rand::UniformReal<float>(-2.0f, 0.0f)(engine) < 0.0f);
    CHECK(alpaka::rand::UniformReal<double>(1e10, 1e10 + 1)(engine) < 1e10 + 1);
}

TEST_CASE("UniformRealUsesTheEngineRange", "[rand]")
{
    // The range of std::minstd_rand.
    using MinstdLike = FixedValueEngine<std::uint32_t, 1, 2'147'483'646>;
    MinstdLike top{2'147'483'646};
    float const topValue = alpaka::rand::UniformReal<float>()(top);
    CHECK(topValue < 1.0f);
    CHECK(topValue > 0.99f);
    MinstdLike bottom{1};
    CHECK(alpaka::math::floatEqualExactNoWarning(alpaka::rand::UniformReal<float>()(bottom), 0.0f));
    // An engine that returns all 64-bit values.
    using Full64 = FixedValueEngine<std::uint64_t, 0, 0xffff'ffff'ffff'ffff>;
    Full64 halfValue{std::uint64_t{1} << 63};
    CHECK(alpaka::math::floatEqualExactNoWarning(alpaka::rand::UniformReal<double>()(halfValue), 0.5));
    Full64 largest{0xffff'ffff'ffff'ffff};
    double const largestValue = alpaka::rand::UniformReal<double>()(largest);
    CHECK(largestValue < 1.0);
    CHECK(largestValue > 0.99);
}

TEST_CASE("HostPhiloxSkipEqualsDiscard", "[rand]")
{
    constexpr std::uint64_t seed = 42;
    constexpr std::uint64_t subsequence = 5;
    auto const reference = drawNumbers(alpaka::rand::Philox4x32x10(seed, subsequence, 0), 48);
    // Each start position in the block, and skips that stay in the block, end it and go past it.
    for(std::size_t draws = 0; draws < 8; ++draws)
    {
        for(std::uint64_t skip = 0; skip <= 17; ++skip)
        {
            PhiloxSingleProbe engine(seed, subsequence, 0);
            for(std::size_t i = 0; i < draws; ++i)
            {
                static_cast<void>(engine());
            }
            engine.skipNumbers(skip);
            INFO("draws " << draws << " skip " << skip);
            CHECK(engine.state.position == (draws + skip) % 4);
            for(std::size_t i = 0; i < 8; ++i)
            {
                CHECK(engine() == reference[draws + skip + i]);
            }
        }
    }
    // A skip of more than 2^32 numbers is the same as the constructor offset.
    for(std::uint64_t draws = 0; draws < 4; ++draws)
    {
        for(std::uint64_t const skip :
            {(std::uint64_t{1} << 34) + 1, (std::uint64_t{1} << 34) + 2, (std::uint64_t{1} << 34) + 3})
        {
            PhiloxSingleProbe engine(seed, subsequence, 0);
            for(std::uint64_t i = 0; i < draws; ++i)
            {
                static_cast<void>(engine());
            }
            engine.skipNumbers(skip);
            alpaka::rand::Philox4x32x10 expected(seed, subsequence, draws + skip);
            INFO("draws " << draws << " skip " << skip);
            for(std::size_t i = 0; i < 8; ++i)
            {
                CHECK(engine() == expected());
            }
        }
    }
}

namespace
{
    // Draws one number and compares it with the expected value, exactly.
    template<typename T, typename TEngine>
    auto drawsExactly(alpaka::rand::UniformReal<T> dist, TEngine engine, T const expected) -> bool
    {
        return alpaka::math::floatEqualExactNoWarning(dist(engine), expected);
    }
} // namespace

TEST_CASE("UniformRealScalesTheUnitValue", "[rand]")
{
    using Full32 = FixedValueEngine<std::uint32_t, 0, 0xffff'ffff>;
    using RealF = alpaka::rand::UniformReal<float>;
    using RealD = alpaka::rand::UniformReal<double>;
    // The unit values 0, 0.5 and the largest float below 1.
    Full32 const zero{0};
    Full32 const halfValue{0x8000'0000};
    Full32 const top{0xffff'ffff};
    constexpr float belowOne = 1.0f - std::numeric_limits<float>::epsilon() / 2;
    CHECK(drawsExactly(RealF(), zero, 0.0f));
    CHECK(drawsExactly(RealF(), halfValue, 0.5f));
    CHECK(drawsExactly(RealF(), top, belowOne));
    CHECK(drawsExactly(RealF(0.0f, 1.0f), halfValue, 0.5f));
    CHECK(drawsExactly(RealF(-0.0f, 1.0f), halfValue, 0.5f));
    CHECK(drawsExactly(RealD(0.0, 1.0), halfValue, 0.5));
    // Ranges that have one limit in common with [0, 1).
    CHECK(drawsExactly(RealF(0.0f, 2.0f), halfValue, 1.0f));
    CHECK(drawsExactly(RealF(0.0f, 0.5f), halfValue, 0.25f));
    CHECK(drawsExactly(RealF(-1.0f, 1.0f), halfValue, 0.0f));
    CHECK(drawsExactly(RealF(0.5f, 1.0f), halfValue, 0.75f));
    CHECK(drawsExactly(RealD(0.0, 2.0), halfValue, 1.0));
    CHECK(drawsExactly(RealD(0.0, 0.5), halfValue, 0.25));
    CHECK(drawsExactly(RealD(-1.0, 1.0), halfValue, 0.0));
    CHECK(drawsExactly(RealD(0.5, 1.0), halfValue, 0.75));
    CHECK(drawsExactly(RealF(-1.0f, 1.0f), zero, -1.0f));
    CHECK(drawsExactly(RealF(0.5f, 1.0f), zero, 0.5f));
    // A vector result uses the same range.
    using Vector = alpaka::rand::Philox4x32x10Vector::ResultContainer<float>;
    alpaka::rand::Philox4x32x10Vector unitEngine(7, 0, 0);
    alpaka::rand::Philox4x32x10Vector rangeEngine(7, 0, 0);
    auto const unit = alpaka::rand::UniformReal<Vector>()(unitEngine);
    auto const range = alpaka::rand::UniformReal<Vector>(0.0f, 2.0f)(rangeEngine);
    for(std::size_t i = 0; i < 4; ++i)
    {
        CHECK(alpaka::math::floatEqualExactNoWarning(range[i], 2.0f * unit[i]));
    }
}

namespace
{
    //! Returns the value through a volatile object: the compiler does not know the result.
    template<typename T>
    auto runTimeValue(T const value) -> T
    {
        T volatile hidden = value;
        return hidden;
    }
} // namespace

TEST_CASE("UniformRealUsesRunTimeLimits", "[rand]")
{
    using Full32 = FixedValueEngine<std::uint32_t, 0, 0xffff'ffff>;
    using RealF = alpaka::rand::UniformReal<float>;
    using RealD = alpaka::rand::UniformReal<double>;
    Full32 const zero{0};
    Full32 const halfValue{0x8000'0000};
    Full32 const top{0xffff'ffff};
    constexpr float belowOne = 1.0f - std::numeric_limits<float>::epsilon() / 2;
    float const zeroF = runTimeValue(0.0f);
    float const oneF = runTimeValue(1.0f);
    // The limits [0, 1) and [-0, 1) give the unit value.
    CHECK(drawsExactly(RealF(zeroF, oneF), zero, 0.0f));
    CHECK(drawsExactly(RealF(zeroF, oneF), halfValue, 0.5f));
    CHECK(drawsExactly(RealF(zeroF, oneF), top, belowOne));
    CHECK(drawsExactly(RealF(runTimeValue(-0.0f), oneF), halfValue, 0.5f));
    CHECK(drawsExactly(RealD(runTimeValue(0.0), runTimeValue(1.0)), halfValue, 0.5));
    // One limit in common with [0, 1).
    CHECK(drawsExactly(RealF(zeroF, runTimeValue(2.0f)), halfValue, 1.0f));
    CHECK(drawsExactly(RealF(runTimeValue(0.5f), oneF), halfValue, 0.75f));
    CHECK(drawsExactly(RealF(runTimeValue(-1.0f), oneF), zero, -1.0f));
    CHECK(drawsExactly(RealD(runTimeValue(0.0), runTimeValue(2.0)), halfValue, 1.0));
    // The product rounds to the upper limit: the result is the largest value below it.
    CHECK(drawsExactly(RealF(runTimeValue(100.0f), runTimeValue(101.0f)), top, std::nextafter(101.0f, 0.0f)));
    CHECK(drawsExactly(RealD(runTimeValue(1e10), runTimeValue(1e10 + 1)), top, std::nextafter(1e10 + 1, 0.0)));
    // A copy of the distribution gives the same numbers.
    RealF const unitRange(zeroF, oneF);
    RealF const otherRange(zeroF, runTimeValue(2.0f));
    RealF const unitCopy(unitRange);
    RealF const otherCopy(otherRange);
    CHECK(drawsExactly(unitCopy, top, belowOne));
    CHECK(drawsExactly(otherCopy, halfValue, 1.0f));
}

//! Draws with distributions that the host made and gave to the kernel, as an argument and as a member.
template<typename T>
class UniformRealArgumentKernel
{
public:
    alpaka::rand::UniformReal<T> m_unitRange;
    alpaka::rand::UniformReal<T> m_otherRange;

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& /* acc */,
        bool* success,
        alpaka::rand::UniformReal<T> unitRange,
        alpaka::rand::UniformReal<T> otherRange,
        T const min,
        T const max) const -> void
    {
        alpaka::rand::Philox4x32x10 engines[6]
            = {{42, 1, 0}, {42, 1, 0}, {42, 1, 0}, {42, 1, 0}, {42, 1, 0}, {42, 1, 0}};
        alpaka::rand::UniformReal<T> reference;
        alpaka::rand::UniformReal<T> local(min, max);
        auto unitMember = m_unitRange;
        auto otherMember = m_otherRange;
        for(int i = 0; i < 64; ++i)
        {
            T const unit = reference(engines[0]);
            T const expected = local(engines[1]);
            ALPAKA_CHECK(*success, expected >= min);
            ALPAKA_CHECK(*success, expected < max);
            ALPAKA_CHECK(*success, alpaka::math::floatEqualExactNoWarning(unitRange(engines[2]), unit));
            ALPAKA_CHECK(*success, alpaka::math::floatEqualExactNoWarning(unitMember(engines[3]), unit));
            ALPAKA_CHECK(*success, alpaka::math::floatEqualExactNoWarning(otherRange(engines[4]), expected));
            ALPAKA_CHECK(*success, alpaka::math::floatEqualExactNoWarning(otherMember(engines[5]), expected));
        }
    }
};

TEMPLATE_LIST_TEST_CASE("UniformRealIsAKernelArgument", "[rand]", TestTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using DataType = std::tuple_element_t<1, TestType>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    DataType const min = runTimeValue(static_cast<DataType>(1.5));
    DataType const max = runTimeValue(static_cast<DataType>(4));
    alpaka::rand::UniformReal<DataType> const unitRange(runTimeValue(DataType{0}), runTimeValue(DataType{1}));
    alpaka::rand::UniformReal<DataType> const otherRange(min, max);
    UniformRealArgumentKernel<DataType> kernel{unitRange, otherRange};

    REQUIRE(fixture(kernel, unitRange, otherRange, min, max));
}

TEST_CASE("UniformRealWithAnInfiniteUpperLimitGivesTheLargestFiniteValue", "[rand]")
{
    using Full32 = FixedValueEngine<std::uint32_t, 0, 0xffff'ffff>;
    Full32 const zero{0};
    Full32 const halfValue{0x8000'0000};
    Full32 const top{0xffff'ffff};
    for(Full32 const engine : {zero, halfValue, top})
    {
        INFO("engine value " << engine.value);
        CHECK(drawsExactly(
            alpaka::rand::UniformReal<float>(0.0f, std::numeric_limits<float>::infinity()),
            engine,
            std::numeric_limits<float>::max()));
        CHECK(drawsExactly(
            alpaka::rand::UniformReal<double>(1.0, std::numeric_limits<double>::infinity()),
            engine,
            std::numeric_limits<double>::max()));
        // The GPU back ends do not support long double in device code.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
        CHECK(drawsExactly(
            alpaka::rand::UniformReal<long double>(0.0L, std::numeric_limits<long double>::infinity()),
            engine,
            std::numeric_limits<long double>::max()));
#endif
    }
}

// The GPU back ends do not support long double in device code.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
TEST_CASE("UniformRealLongDoubleStaysBelowTheUpperLimit", "[rand]")
{
    using RealL = alpaka::rand::UniformReal<long double>;
    // The largest unit value of a 64-bit engine: the product rounds to the upper limit.
    FixedValueEngine<std::uint64_t, 0, 0xffff'ffff'ffff'ffff> top{0xffff'ffff'ffff'ffff};
    long double const positive = RealL(1.0L, 2.0L)(top);
    CHECK(positive >= 1.0L);
    CHECK(positive < 2.0L);
    long double const negative = RealL(-2.0L, -1.0L)(top);
    CHECK(negative >= -2.0L);
    CHECK(negative < -1.0L);
    long double const zeroLimit = RealL(-1.0L, 0.0L)(top);
    CHECK(zeroLimit >= -1.0L);
    CHECK(zeroLimit < 0.0L);
    long double const large = RealL(1.0e30L, 2.0e30L)(top);
    CHECK(large >= 1.0e30L);
    CHECK(large < 2.0e30L);
}

TEST_CASE("UniformRealLongDoubleStaysAboveTheLowerLimit", "[rand]")
{
    using RealL = alpaka::rand::UniformReal<long double>;
    FixedValueEngine<std::uint64_t, 0, 0xffff'ffff'ffff'ffff> top{0xffff'ffff'ffff'ffff};
    // The range has one value: the limits are neighbours.
    long double const belowOne = std::nextafter(1.0L, 0.0L);
    long double const neighbour = RealL(belowOne, 1.0L)(top);
    CHECK(neighbour >= belowOne);
    CHECK(neighbour < 1.0L);
    long double const belowThree = std::nextafter(3.0L, 0.0L);
    long double const neighbourOfThree = RealL(belowThree, 3.0L)(top);
    CHECK(neighbourOfThree >= belowThree);
    CHECK(neighbourOfThree < 3.0L);
    // Subnormal limits.
    long double const tiny = std::numeric_limits<long double>::denorm_min();
    long double const subnormal = RealL(2 * tiny, 3 * tiny)(top);
    CHECK(subnormal >= 2 * tiny);
    CHECK(subnormal < 3 * tiny);
    long double const negativeSubnormal = RealL(-4 * tiny, -3 * tiny)(top);
    CHECK(negativeSubnormal >= -4 * tiny);
    CHECK(negativeSubnormal < -3 * tiny);
}
#endif
