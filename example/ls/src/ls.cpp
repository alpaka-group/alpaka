// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: ISC

#include "alpaka/test/acc/TestAccs.hpp"

#include <alpaka/alpaka.hpp>

struct PerAcc
{
    template<typename TAcc>
    void operator()() const
    {
        auto const platform = alpaka::Platform<TAcc>{};
        std::cout << alpaka::getAccName<TAcc>() << '\n';
        for(auto const& dev : alpaka::getDevs(platform))
            std::cout << '\t' << alpaka::getName(dev) << '\n';
    }
};

int main()
{
    using Idx = int;
    using Dim = alpaka::DimInt<1>;
    alpaka::meta::forEachType<alpaka::test::EnabledAccs<Dim, Idx>>(PerAcc{});
}
