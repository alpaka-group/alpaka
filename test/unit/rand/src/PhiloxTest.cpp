/* Copyright 2024 Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/rand/RandPhilox.hpp>
#include <alpaka/rand/RandPhiloxStateless.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
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
