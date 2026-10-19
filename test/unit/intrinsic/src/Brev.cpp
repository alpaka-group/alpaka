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
class BrevTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        TInput const inputs[] = {0, 1, 3, 64, 256, 51362, std::numeric_limits<TInput>::max()};
        for(auto const input : inputs)
        {
            TInput const expected = brevNaive(input);
            TInput const actual = alpaka::brev(acc, input);
            ALPAKA_CHECK(*success, actual == expected);
        }
    }

private:
    ALPAKA_FN_ACC static auto brevNaive(TInput value) -> TInput
    {
        if(value == 0 || value == std::numeric_limits<TInput>::max())
            return value;

        TInput mask{1};
        TInput res{0};

        constexpr std::uint32_t bits = sizeof(value) * 8;
        for(std::uint32_t bit = 0; bit < bits; bit++)
        {
            if(value & mask)
            {
                res |= mask << (bits - bit - 1);
            }
            mask = mask << 1;
        }

        return res;
    }
};

TEMPLATE_LIST_TEST_CASE("brev", "[intrinsic]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    BrevTestKernel<std::uint32_t> kernel32bit;
    REQUIRE(fixture(kernel32bit));

    BrevTestKernel<std::uint64_t> kernel64bit;
    REQUIRE(fixture(kernel64bit));
}
