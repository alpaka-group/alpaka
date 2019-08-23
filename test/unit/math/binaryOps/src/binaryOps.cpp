//
// Created by jakob on 19.08.19.
//

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <iostream>
#include <typeinfo>
#include <cmath>

/*
 * atan2     | Y | R^2\{(0,0)}
 * fmod      | Y | R^2\{(x,0)|x in R}
 * max       | Y | R^2
 * min       | Y | R^2
 * remainder | Y | R^2\{(x,0)|x in R}
 * sincos    | Y | R^2 -> return 2 R
*/

/*
 * if you want to add a new operation simply add it to the array.
 * 1. Specify the std::  implementation.
 * 2. If the std function has a const reference signature use this
 * 3. Specify the alpaka implementation.
 * 4. Define the range in which the operator should be testes against.
 */

/*
 * if you need to add a new range you have to add it to the switch case
 *  - in the kernel class
 *  - in the TestTemplate
 */




// possible definition ranges
enum class Range
{
    POSITIVE_ONLY,
    NOT_ZER0,
    X_NOT_ZERO,
    Y_NOT_ZERO,
    UNRESTRICTED
};

// function pointer templates for std::math and alpaka::math
template<
        typename TAcc,
        typename T>
using alpaka_func_ptr = T (*) (TAcc const & , T const &, T const &);

template<
        typename T>
using std_func_ptr = T (*) (T, T);

// used for all operators that need const references (like min())
template<
        typename T>
using std_func_ptr_const = T const & (*) (T const & , T const &);


template<
        typename TAcc,
        typename T>
struct TestStruct
{
    std_func_ptr<T> stdOp ;
    std_func_ptr_const <T> stdAlternative;
    alpaka_func_ptr<TAcc, T> alpakaOp;
    Range range;
};


class BinaryOpsKernel{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<
            typename TAcc,
            typename TData,
            typename TIdx
    >
    ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TIdx const & numOps,
            TIdx const & sizeArgs,
            TData const * const argsX,
            TData const * const argsY,
            TestStruct<TAcc, TData> const * const structs,
            TData * results
    ) const
    -> void
    {
        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TData argX;
        TData argY;
        if(gridThreadIdx < numOps)
        {
            // sizeRes = numOps * sizeArgs
            switch (structs[gridThreadIdx].range)
            {
                case Range::POSITIVE_ONLY:
                    for(TIdx i(0); i < sizeArgs/2 -1; ++i)
                    {
                        argX = argsX[i];
                        argY = argsY[i];
                        results[i + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, argX, argY);
                    }
                    break;
                case Range::NOT_ZER0:
                    for(TIdx i(0); i < sizeArgs; ++i)
                    {
                        if(i == sizeArgs/2 -1 || i == sizeArgs/2)
                            continue;
                        argX = argsX[i];
                        argY = argsY[i];
                        results[i + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, argX, argY);
                    }
                    break;

                case Range::X_NOT_ZERO:
                    for(TIdx i(0); i < sizeArgs/2 -1; ++i)
                    {
                        if(i == sizeArgs/2 -1 || i == sizeArgs/2)
                            argX = argsX[0];
                        else
                            argX = argsX[i];
                        argY = argsY[i];
                        results[i + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, argX, argY);
                    }
                    break;

                case Range::Y_NOT_ZERO:
                    for(TIdx i(0); i < sizeArgs/2 -1; ++i)
                    {
                        if(i == sizeArgs/2 -1 || i == sizeArgs/2)
                            argY = argsY[0];
                        else
                            argY = argsY[i];
                        argX = argsX[i];

                        results[i + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, argX, argY);
                    }
                    break;

                case Range::UNRESTRICTED:
                    for(TIdx i(0); i < sizeArgs; ++i)
                    {
                        argX = argsX[i];
                        argY= argsY[i];
                        results[i + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, argX, argY);
                    }
                    break;

                default:
                    break;
            }
        }
    }
};


struct TestTemplate {
    template<typename TAcc>
    void operator()() {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using Data = double;
        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;
        using PltfHost = alpaka::pltf::PltfCpu;


        std::cout << "\nTesting next AccType \n\n";
        // the functions that will be tested


        TestStruct<TAcc, Data> arr[] =
                {/*     normal callback,    const callback, alpaka callback,                           definition range*/
                        { &std::atan2,       nullptr,        &alpaka::math::atan2<TAcc, Data, Data>,     Range::NOT_ZER0},
                        { &std::fmod,        nullptr,        &alpaka::math::fmod<TAcc, Data, Data>,      Range::Y_NOT_ZERO},
                        { nullptr,           &std::max,      &alpaka::math::max<TAcc, Data, Data>,       Range::Y_NOT_ZERO},
                        { nullptr,           &std::min,      &alpaka::math::min<TAcc, Data, Data>,       Range::UNRESTRICTED},
                        { &std::pow,         nullptr,        &alpaka::math::pow<TAcc, Data, Data>,       Range::POSITIVE_ONLY},
                        { &std::remainder,   nullptr,        &alpaka::math::remainder<TAcc, Data, Data>, Range::POSITIVE_ONLY}
                };

        Idx const numOps = sizeof(arr) / sizeof(TestStruct<TAcc, Data>);
        Idx const elementsPerThread(1u);
        Idx const sizeArgs(10u);
        Idx const sizeRes = sizeArgs * numOps;

        // Create the kernel function object.
        BinaryOpsKernel kernel;

        // Get the host device.
        auto const devHost(
                alpaka::pltf::getDevByIdx<PltfHost>(0u));

        // Select a device to execute on.
        auto const devAcc(
                alpaka::pltf::getDevByIdx<PltfAcc>(0u));

        // Get a queue on this device.
        QueueAcc queue(devAcc);

        alpaka::vec::Vec<Dim, Idx> const extent(numOps);

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
                alpaka::workdiv::getValidWorkDiv<TAcc>(
                        devAcc,
                        extent,
                        elementsPerThread,
                        false,
                        alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));


        // Allocate host memory buffers.
        auto memBufHostArgsX(alpaka::mem::buf::alloc<Data, Idx>(devHost, sizeArgs));
        auto memBufHostArgsY(alpaka::mem::buf::alloc<Data, Idx>(devHost, sizeArgs));
        auto memBufHostRes(alpaka::mem::buf::alloc<Data, Idx>(devHost, sizeRes));
        auto memBufHostStructs(alpaka::mem::buf::alloc<TestStruct<TAcc, Data>, Idx>(devHost, extent));

        Data *const pBufHostArgsX = alpaka::mem::view::getPtrNative(memBufHostArgsX);
        Data *const pBufHostArgsY = alpaka::mem::view::getPtrNative(memBufHostArgsY);
        Data *const pBufHostRes = alpaka::mem::view::getPtrNative(memBufHostRes);
        TestStruct<TAcc, Data> *const pBufHostStructs = alpaka::mem::view::getPtrNative(memBufHostStructs);

        // This is just for a better understanding which results are unchanged.
        for (Idx i(0); i < numOps; ++i)
        {
            for (Idx j(0); j < sizeArgs; ++j) {
                pBufHostRes[j + i * sizeArgs] = -1;
            }
        }

        // C++11 random generator for uniformly distributed numbers in [-100, 100].
        std::random_device rd{};
        std::default_random_engine eng{rd()};
        std::uniform_real_distribution<Data> dist(0, 100);

        // Initiate the arguments.
        for (Idx i(0); i < sizeArgs / 2 - 1; ++i)
        {
            pBufHostArgsX[i] = dist(eng);
            pBufHostArgsY[i] = dist(eng);

        }
        pBufHostArgsX[sizeArgs / 2 - 1] = 0.0;
        pBufHostArgsY[sizeArgs / 2 - 1] = 0.0;

        pBufHostArgsX[sizeArgs / 2] = -0.0;
        pBufHostArgsY[sizeArgs / 2] = -0.0;

        for (Idx i(sizeArgs / 2 + 1); i < sizeArgs; ++i)
        {
            pBufHostArgsX[i] = dist(eng) - 100;
            pBufHostArgsY[i] = dist(eng) - 100;
        }


        // Initiate the structs.
        for (Idx i(0u); i < numOps; ++i)
        {
            pBufHostStructs[i] = arr[i];
        }


        // Allocate the buffer on the accelerator.
        auto memBufAccArgsX(alpaka::mem::buf::alloc<Data, Idx>(devAcc, sizeArgs));
        auto memBufAccArgsY(alpaka::mem::buf::alloc<Data, Idx>(devAcc, sizeArgs));
        auto memBufAccRes(alpaka::mem::buf::alloc<Data, Idx>(devAcc, sizeRes));
        auto memBufAccStructs(alpaka::mem::buf::alloc<TestStruct<TAcc, Data>, Idx>(devAcc, numOps));


        // Copy Host -> Acc.
        alpaka::mem::view::copy(queue, memBufAccArgsX, memBufHostArgsX, sizeArgs);
        alpaka::mem::view::copy(queue, memBufAccArgsY, memBufHostArgsY, sizeArgs);
        alpaka::mem::view::copy(queue, memBufAccRes, memBufHostRes, sizeRes);
        alpaka::mem::view::copy(queue, memBufAccStructs, memBufHostStructs, numOps);

        for (Idx i(0u); i < sizeArgs; ++i) {
            std::cout << "bufferArgs x: " << pBufHostArgsX[i] << " y: " << pBufHostArgsY[i] << std::endl;
        }

        auto pMemBufAccArgsX = alpaka::mem::view::getPtrNative(memBufAccArgsX);
        auto pMemBufAccArgsY = alpaka::mem::view::getPtrNative(memBufAccArgsY);
        auto pMemBufAccRes = alpaka::mem::view::getPtrNative(memBufAccRes);
        auto pMemBufAccStructs = alpaka::mem::view::getPtrNative(memBufAccStructs);



        // Create the kernel execution task.
        auto const taskKernel(alpaka::kernel::createTaskKernel<TAcc>(
                workDiv,
                kernel,
                numOps,
                sizeArgs,
                pMemBufAccArgsX,
                pMemBufAccArgsY,
                pMemBufAccStructs,
                pMemBufAccRes
        ));

        // Enqueue the kernel execution task.
        alpaka::queue::enqueue(queue, taskKernel);

        // Copy back the result.
        alpaka::mem::view::copy(queue, memBufHostRes, memBufAccRes, sizeRes);


        // Wait for the queue to finish the memory operation.
        alpaka::wait::wait(queue);


        // Print out all results.
        for (Idx i(0); i < numOps; ++i)
        {
            std::cout << "\nResults " << i + 1 << ". function:\n";

            for (Idx j(0); j < sizeArgs; ++j)
            {
                Data const &res(pBufHostRes[j + i * sizeArgs]);
                std::cout << "bufferResults: " << res << "\n";
            }
        }

        // Check device result against host result.

        Data argX;
        Data argY;
        Data stdRes;
        TestStruct<TAcc, Data> t;
        for (Idx j(0); j < numOps; ++j)
        {
            t = arr[j];
            switch (t.range)
            {
                case Range::POSITIVE_ONLY:
                    for (Idx i(0); i < sizeArgs / 2 - 1; ++i)
                    {
                        argX = pBufHostArgsX[i];
                        argY = pBufHostArgsY[i];
                        if(t.stdOp != nullptr)
                            stdRes = t.stdOp(argX, argY);
                        else
                            stdRes = t.stdAlternative(argX, argY);
                        REQUIRE(stdRes == Approx(pBufHostRes[i + sizeArgs * j]));
                    }
                    break;
                case Range::NOT_ZER0:
                    for (Idx i(0); i < sizeArgs; ++i)
                    {
                        if (i == sizeArgs / 2 - 1 || i == sizeArgs / 2)
                            continue;
                        argX = pBufHostArgsX[i];
                        argY = pBufHostArgsY[i];
                        if(t.stdOp != nullptr)
                            stdRes = t.stdOp(argX, argY);
                        else
                            stdRes = t.stdAlternative(argX, argY);
                        REQUIRE(stdRes == Approx(pBufHostRes[i + sizeArgs * j]));
                    }
                    break;

                case Range::X_NOT_ZERO:
                    for (Idx i(0); i < sizeArgs / 2 - 1; ++i)
                    {
                        if (i == sizeArgs / 2 - 1 || i == sizeArgs / 2)
                            argX = pBufHostArgsX[0];
                        else
                            argX = pBufHostArgsX[i];
                        argY = pBufHostArgsY[i];
                        if(t.stdOp != nullptr)
                            stdRes = t.stdOp(argX, argY);
                        else
                            stdRes = t.stdAlternative(argX, argY);
                        REQUIRE(stdRes == Approx(pBufHostRes[i + sizeArgs * j]));
                    }
                    break;

                case Range::Y_NOT_ZERO:
                    for (Idx i(0); i < sizeArgs / 2 - 1; ++i)
                    {
                        if (i == sizeArgs / 2 - 1 || i == sizeArgs / 2)
                            argY = pBufHostArgsY[0];
                        else
                            argY = pBufHostArgsY[i];
                        argX = pBufHostArgsX[i];
                        if(t.stdOp != nullptr)
                            stdRes = t.stdOp(argX, argY);
                        else
                            stdRes = t.stdAlternative(argX, argY);
                        REQUIRE(stdRes == Approx(pBufHostRes[i + sizeArgs * j]));
                    }
                    break;

                case Range::UNRESTRICTED:
                    for (Idx i(0); i < sizeArgs; ++i)
                    {
                        argX = pBufHostArgsX[i];
                        argY = pBufHostArgsY[i];
                        if(t.stdOp != nullptr)
                            stdRes = t.stdOp(argX, argY);
                        else
                            stdRes = t.stdAlternative(argX, argY);
                        REQUIRE(stdRes == Approx(pBufHostRes[i + sizeArgs * j]));
                    }
                    break;

                default:
                    break;
            }
        }
    }
};

TEST_CASE("binaryOps", "[binaryOps]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
            alpaka::dim::DimInt<1u>,
            std::size_t>;

    alpaka::meta::forEachType< TestAccs >( TestTemplate() );
}