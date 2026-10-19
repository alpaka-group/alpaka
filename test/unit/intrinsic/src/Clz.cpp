/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/intrinsic/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

template<typename TInput>
class ClzTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        TInput const inputs[] = {0, 1, 3, 64, 256, 51362, std::numeric_limits<TInput>::max()};
        for(auto const input : inputs)
        {
            std::int32_t const expected = clzNaive(input);
            std::int32_t const actual = alpaka::clz(acc, input);
            ALPAKA_CHECK(*success, actual == expected);
        }
    }

private:
    ALPAKA_FN_ACC static auto clzNaive(TInput value) -> std::int32_t
    {
        if(value == 0)
            return sizeof(TInput) == 8 ? 32 : 64;
        else if(value == std::numeric_limits<TInput>::max())
            return 0;
        std::int32_t result = 0;

        TInput mask = 1;

        for(std::uint32_t i = 0; i < sizeof(TInput) * 8; i++)
        {
            if((value & mask) == 0)
            {
                result += 1;
            }
            mask = mask << 1;
        }
        return result;
    }
};

TEMPLATE_LIST_TEST_CASE("clz", "[intrinsic]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    ClzTestKernel<std::uint32_t> kernel32bit;
    REQUIRE(fixture(kernel32bit));

    ClzTestKernel<std::uint64_t> kernel64bit;
    REQUIRE(fixture(kernel64bit));
}
