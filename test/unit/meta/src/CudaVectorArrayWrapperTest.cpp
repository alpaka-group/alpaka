/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Unused.hpp>
#    include <alpaka/meta/CudaVectorArrayWrapper.hpp>
#    include <alpaka/meta/IsStrictBase.hpp>
#    include <alpaka/rand/Traits.hpp>
#    include <alpaka/test/KernelExecutionFixture.hpp>
#    include <alpaka/test/acc/TestAccs.hpp>

#    include <catch2/catch.hpp>

#    include <type_traits>


template<typename T>
class CudaVectorArrayWrapperTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        alpaka::ignore_unused(acc);

        using T1 = alpaka::meta::CudaVectorArrayWrapper<T, 1>;
        T1 t1{0};
        static_assert(T1::size == 1, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size<T1>::value == 1, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same<decltype(t1[0]), T&>::value, "CudaVectorArrayWrapper in-kernel type test failed!");
#    ifdef __GNUC__
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
        ALPAKA_CHECK(*success, t1[0] == 0);
#    ifdef __GNUC__
#        pragma GCC diagnostic pop
#    endif

        using T2 = alpaka::meta::CudaVectorArrayWrapper<T, 2>;
        T2 t2{0, 1};
        static_assert(T2::size == 2, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size<T2>::value == 2, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same<decltype(t2[0]), T&>::value, "CudaVectorArrayWrapper in-kernel type test failed!");
#    ifdef __GNUC__
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
        ALPAKA_CHECK(*success, t2[0] == 0);
        ALPAKA_CHECK(*success, t2[1] == 1);
#    ifdef __GNUC__
#        pragma GCC diagnostic pop
#    endif

        using T3 = alpaka::meta::CudaVectorArrayWrapper<T, 3>;
        T3 t3{0, 0, 0};
        t3 = {0, 1, 2};
        static_assert(T3::size == 3, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size<T3>::value == 3, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same<decltype(t3[0]), T&>::value, "CudaVectorArrayWrapper in-kernel type test failed!");
#    ifdef __GNUC__
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
        ALPAKA_CHECK(*success, t3[0] == 0);
        ALPAKA_CHECK(*success, t3[1] == 1);
        ALPAKA_CHECK(*success, t3[2] == 2);
#    ifdef __GNUC__
#        pragma GCC diagnostic pop
#    endif

        using T4 = alpaka::meta::CudaVectorArrayWrapper<T, 4>;
        T4 t4{0, 0, 0, 0};
        t4[1] = 1;
        t4[2] = t4[1] + 1;
        t4[3] = t4[2] + t2[1];
        static_assert(T4::size == 4, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size<T4>::value == 4, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same<decltype(t4[0]), T&>::value, "CudaVectorArrayWrapper in-kernel type test failed!");
#    ifdef __GNUC__
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
        ALPAKA_CHECK(*success, t4[0] == 0);
        ALPAKA_CHECK(*success, t4[1] == 1);
        ALPAKA_CHECK(*success, t4[2] == 2);
        ALPAKA_CHECK(*success, t4[3] == 3);
#    ifdef __GNUC__
#        pragma GCC diagnostic pop
#    endif
    }
};

TEMPLATE_LIST_TEST_CASE("cudaVectorArrayWrapperDevice", "[meta]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    CudaVectorArrayWrapperTestKernel<int> kernelInt;
    REQUIRE(fixture(kernelInt));

    CudaVectorArrayWrapperTestKernel<unsigned> kernelUnsigned;
    REQUIRE(fixture(kernelUnsigned));

    CudaVectorArrayWrapperTestKernel<float> kernelFloat;
    REQUIRE(fixture(kernelFloat));

    CudaVectorArrayWrapperTestKernel<double> kernelDouble;
    REQUIRE(fixture(kernelDouble));
}


TEST_CASE("cudaVectorArrayWrapperHost", "[meta]")
{
    // TODO: It would be nice to check all possible type vs. size combinations.

    using Float1 = alpaka::meta::CudaVectorArrayWrapper<float, 1>;
    Float1 floatWrapper1{-1.0f};
    STATIC_REQUIRE(Float1::size == 1);
    STATIC_REQUIRE(std::tuple_size<Float1>::value == 1);
    STATIC_REQUIRE(std::is_same<decltype(floatWrapper1[0]), float&>::value);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<float1, Float1>::value);
#    ifdef __GNUC__
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
    REQUIRE(floatWrapper1[0] == -1.0f);
#    ifdef __GNUC__
#        pragma GCC diagnostic pop
#    endif

    using Int1 = alpaka::meta::CudaVectorArrayWrapper<int, 1>;
    Int1 intWrapper1 = {-42};
    STATIC_REQUIRE(Int1::size == 1);
    STATIC_REQUIRE(std::tuple_size<Int1>::value == 1);
    STATIC_REQUIRE(std::is_same<decltype(intWrapper1[0]), int&>::value);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<int1, Int1>::value);
    REQUIRE(intWrapper1[0] == -42);

    using Uint2 = alpaka::meta::CudaVectorArrayWrapper<unsigned, 2>;
    Uint2 uintWrapper2{0u, 1u};
    STATIC_REQUIRE(Uint2::size == 2);
    STATIC_REQUIRE(std::tuple_size<Uint2>::value == 2);
    STATIC_REQUIRE(std::is_same<decltype(uintWrapper2[0]), unsigned&>::value);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<uint2, Uint2>::value);
    REQUIRE(uintWrapper2[0] == 0u);
    REQUIRE(uintWrapper2[1] == 1u);

    using Uint4 = alpaka::meta::CudaVectorArrayWrapper<unsigned, 4>;
    Uint4 uintWrapper4{0u, 0u, 0u, 0u};
    STATIC_REQUIRE(Uint4::size == 4);
    STATIC_REQUIRE(std::tuple_size<Uint4>::value == 4);
    STATIC_REQUIRE(std::is_same<decltype(uintWrapper4[0]), unsigned&>::value);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<uint4, Uint4>::value);
    uintWrapper4[1] = 1u;
    uintWrapper4[2] = uintWrapper4[1] + 1u;
    uintWrapper4[3] = uintWrapper4[2] + uintWrapper2[1];
    REQUIRE(uintWrapper4[0] == 0u);
    REQUIRE(uintWrapper4[1] == 1u);
    REQUIRE(uintWrapper4[2] == 2u);
    REQUIRE(uintWrapper4[3] == 3u);

    using Double3 = alpaka::meta::CudaVectorArrayWrapper<double, 3>;
    Double3 doubleWrapper3{0.0, 0.0, 0.0};
    doubleWrapper3 = {0.0, -1.0, -2.0};
    STATIC_REQUIRE(Double3::size == 3);
    STATIC_REQUIRE(std::tuple_size<Double3>::value == 3);
    STATIC_REQUIRE(std::is_same<decltype(doubleWrapper3[0]), double&>::value);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<double3, Double3>::value);
#    ifdef __GNUC__
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
    REQUIRE(doubleWrapper3[0] == 0.0);
    REQUIRE(doubleWrapper3[1] == -1.0);
    REQUIRE(doubleWrapper3[2] == -2.0);
#    ifdef __GNUC__
#        pragma GCC diagnostic pop
#    endif
}

#endif
