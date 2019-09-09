/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "../../dataGen.hpp"
#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <iostream>

/** List of all operators
 * operator  | in std | definition |  range | notes
 * abs       | Y | R
 * acos      | Y | [-1, 1]
 * asin      | Y | [-1, 1]
 * atan      | Y | R
 * cbrt      | Y | R | third root of arg
 * ceil      | Y | R
 * cos       | Y | R
 * erf       | Y | R | error function for arg
 * exp       | Y | R | e^arg
 * floor     | Y | R
 * log       | Y | N\{0}
 * round     | Y | R
 * rsqrt     | X | N\{0} | inverse square root
 * sin       | Y | R
 * sqrt      | Y | N
 * tan       | Y | [x | x \= pi/2 + k*pi, k in Z]
 * trunc     | Y | R | round towards zero
 *
 * If you want to add a new operation simply add it to the array.
 * 1. Specify the std::  implementation.
 * 2. Specify the alpaka implementation.
 * 3. Define the range in which the operator should be testes against.
 *
 * If you need to add a new range you have to add it to the switch case
 *  - in the kernel class
 *  - in the TestTemplate
 *
 *  If the wrong range is used,
 *  the bahaviour depends on the individual implementation.
 */

// Custom functions.
template<typename T>
T rsqrt(T t){
    return 1 / std::sqrt(t);
}

//! @enum Range
//! @brief Possible definition ranges.
enum class Range
{
    POSITIVE_ONLY,
    POSITIVE_AND_ZERO,
    ONE_NEIGHBOURHOOD, // [-1, 1]
    UNRESTRICTED
};

// C-Style Callbacks for std::math and alpaka::math.
template<
    typename TAcc,
    typename T>
using alpaka_func_ptr = T (*) (TAcc const & , T const &);

template<
    typename T>
using std_func_ptr = T (*) (T);

/*!
 * @struct TestStruct
 * @tparam TAcc The type of the used accelerator, important for alpaka callback.
 * @tparam T The data type (float || double).
 * @var stdOps
 * @brief Callback the std implementation.
 * @var alpakaOp
 * @brief Callback to alpaka implementation.
 * @note Behaviour for nullptr is undefined.
 */
template<
    typename TAcc,
    typename T>
struct TestStruct
{
    std_func_ptr< T > stdOp;
    alpaka_func_ptr<
        TAcc,
        T
    > alpakaOp;
    Range range;
};

class UnaryOpsKernel
{
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
        TData const * const args,
        TestStruct<TAcc, TData> const * const structs,
        TData * results
        ) const
        -> void
    {
        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TData arg;
        if(gridThreadIdx < numOps)
        {
            switch (structs[gridThreadIdx].range)
            {
                case Range::POSITIVE_ONLY:
                    for(TIdx row(0); row < sizeArgs / 2 - 1; ++row)
                    {
                        arg = args[row];
                        results[row + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    }
                    break;
                case Range::POSITIVE_AND_ZERO:
                    for(TIdx row(0); row < sizeArgs / 2; ++row)
                    {
                        arg = args[row];
                        results[row + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    }
                    break;
                case Range::UNRESTRICTED:
                    for(TIdx row(0); row < sizeArgs; ++row)
                    {
                        arg = args[row];
                        results[row + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    }
                    break;
                case Range::ONE_NEIGHBOURHOOD:
                    // this option is hardcoded because it is to specific
                    if(sizeArgs < 4)
                        break;
                    arg = 1;
                    results[0 + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    arg = 0.5;
                    results[1 + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    arg = 0;
                    results[2 + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    arg = -0.5;
                    results[3 + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    arg = -1;
                    results[4 + gridThreadIdx* sizeArgs] = structs[gridThreadIdx].alpakaOp(acc, arg);
                    break;

                default:
                    break;
            }

        }
    }
};

template <typename Data>
struct TestTemplate
{
public:
    template< typename TAcc >
    void operator()( )
    {
        //using Data = double; // this can be changed to float without problems
        using Dim = alpaka::dim::Dim< TAcc >;
        using Idx = alpaka::idx::Idx< TAcc >;
        using DevAcc = alpaka::dev::Dev< TAcc >;
        using PltfAcc = alpaka::pltf::Pltf< DevAcc >;
        using QueueAcc = alpaka::test::queue::DefaultQueue< DevAcc >;
        using PltfHost = alpaka::pltf::PltfCpu;
        // the functions that will be tested

        TestStruct<TAcc, Data> arr [] =
            {
                { &std::abs,    &alpaka::math::abs<TAcc,Data>,      Range::UNRESTRICTED      },
                { &std::acos,   &alpaka::math::acos<TAcc,Data>,     Range::ONE_NEIGHBOURHOOD },
                { &std::asin,   &alpaka::math::asin<TAcc,Data>,     Range::ONE_NEIGHBOURHOOD },
                { &std::atan,   &alpaka::math::atan<TAcc,Data>,     Range::UNRESTRICTED      },
                { &std::cbrt,   &alpaka::math::cbrt<TAcc,Data>,     Range::UNRESTRICTED      },
                { &std::ceil,   &alpaka::math::ceil<TAcc,Data>,     Range::UNRESTRICTED      },
                { &std::cos,    &alpaka::math::cos<TAcc,Data>,      Range::UNRESTRICTED      },
                { &std::erf,    &alpaka::math::erf<TAcc,Data>,      Range::UNRESTRICTED      },
                { &std::exp,    &alpaka::math::exp<TAcc,Data>,      Range::UNRESTRICTED      },
                { &std::floor,  &alpaka::math::floor<TAcc,Data>,    Range::UNRESTRICTED      },
                { &std::log,    &alpaka::math::log<TAcc,Data>,      Range::POSITIVE_ONLY     },
                { &std::round,  &alpaka::math::round<TAcc,Data>,    Range::UNRESTRICTED      },
                { &rsqrt<Data>, &alpaka::math::rsqrt<TAcc,Data>,    Range::POSITIVE_ONLY     },
                { &std::sin,    &alpaka::math::sin<TAcc,Data>,      Range::UNRESTRICTED      },
                { &std::sqrt,   &alpaka::math::sqrt<TAcc,Data>,     Range::POSITIVE_AND_ZERO },
                { &std::tan,    &alpaka::math::tan<TAcc,Data>,      Range::UNRESTRICTED      },
                { &std::trunc,  &alpaka::math::trunc<TAcc,Data>,    Range::UNRESTRICTED      }
            };

        Idx const
            numOps =
            sizeof( arr ) / sizeof( TestStruct< TAcc, Data > );
        Idx const elementsPerThread( 1u );
        // sizeArgs defines how many test arguments are used.
        // It should be an even number greater than 4.
        Idx const sizeArgs( 100u );
        Idx const sizeRes = sizeArgs * numOps;
        constexpr size_t randomRange = 100u;


        // Create the kernel function object.
        UnaryOpsKernel kernel;

        // Get the host device.
        auto const devHost(
            alpaka::pltf::getDevByIdx< PltfHost >( 0u )
        );

        // Select a device to execute on.
        auto const devAcc(
            alpaka::pltf::getDevByIdx< PltfAcc >( 0u )
        );

        // Get a queue on this device.
        QueueAcc queue( devAcc );

        alpaka::vec::Vec<
            Dim,
            Idx
        > const extent( numOps );

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<
            Dim,
            Idx
        > const workDiv(
            alpaka::workdiv::getValidWorkDiv< TAcc >(
                devAcc,
                extent,
                elementsPerThread,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
            )
        );


        // Allocate host memory buffers.
        auto
            memBufHostArgs
            (
                alpaka::mem::buf::alloc<
                    Data,
                    Idx
                >(
                    devHost,
                    sizeArgs
                )
            );
        auto
            memBufHostRes
            (
                alpaka::mem::buf::alloc<
                    Data,
                    Idx
                >(
                    devHost,
                    sizeRes
                )
            );
        auto
            memBufHostStructs
            (
                alpaka::mem::buf::alloc<
                    TestStruct<
                        TAcc,
                        Data
                    >,
                    Idx
                >(
                    devHost,
                    extent
                )
            );

        Data
            * const pBufHostArgs =
            alpaka::mem::view::getPtrNative( memBufHostArgs );
        Data
            * const pBufHostRes =
            alpaka::mem::view::getPtrNative( memBufHostRes );
        TestStruct<
            TAcc,
            Data
        >
            * const pBufHostStructs =
            alpaka::mem::view::getPtrNative( memBufHostStructs );

        // This is just for a better understanding which results are unchanged.
        for( Idx i( 0 ); i < numOps; ++i )
        {
            for( Idx j( 0 ); j < sizeArgs; ++j )
            {
                pBufHostRes[j + i * sizeArgs] = -1;
            }
        }

        unsigned long seed =
            test::fillWithRndArgs< Data >(pBufHostArgs, sizeArgs, randomRange);
        std::cout << "Using seed: " << seed <<std::endl;


        // Initiate the structs.
        for( Idx i( 0u ); i < numOps; ++i )
        {
            pBufHostStructs[i] = arr[i];
        }

        // Allocate the buffer on the accelerator.
        auto
            memBufAccArgs
            (
                alpaka::mem::buf::alloc<
                    Data,
                    Idx
                >(
                    devAcc,
                    sizeArgs
                )
            );
        auto
            memBufAccRes
            (
                alpaka::mem::buf::alloc<
                    Data,
                    Idx
                >(
                    devAcc,
                    sizeRes
                )
            );
        auto
            memBufAccStructs
            (
                alpaka::mem::buf::alloc<
                    TestStruct<
                        TAcc,
                        Data
                    >,
                    Idx
                >(
                    devAcc,
                    numOps
                )
            );


        // Copy Host -> Acc.
        alpaka::mem::view::copy(
            queue,
            memBufAccArgs,
            memBufHostArgs,
            sizeArgs
        );
        alpaka::mem::view::copy(
            queue,
            memBufAccRes,
            memBufHostRes,
            sizeRes
        );
        alpaka::mem::view::copy(
            queue,
            memBufAccStructs,
            memBufHostStructs,
            numOps
        );

        auto pMemBufAccArgs = alpaka::mem::view::getPtrNative( memBufAccArgs );
        auto pMemBufAccRes = alpaka::mem::view::getPtrNative( memBufAccRes );
        auto
            pMemBufAccStructs =
            alpaka::mem::view::getPtrNative( memBufAccStructs );


        // Create the kernel execution task.
        auto const taskKernel(
            alpaka::kernel::createTaskKernel< TAcc >(
                workDiv,
                kernel,
                numOps,
                sizeArgs,
                pMemBufAccArgs,
                pMemBufAccStructs,
                pMemBufAccRes
            )
        );

        // Enqueue the kernel execution task.
        alpaka::queue::enqueue(
            queue,
            taskKernel
        );

        // Copy back the result.
        alpaka::mem::view::copy(
            queue,
            memBufHostArgs,
            memBufAccArgs,
            sizeArgs
        );
        alpaka::mem::view::copy(
            queue,
            memBufHostRes,
            memBufAccRes,
            sizeRes
        );


        // Wait for the queue to finish the memory operation.
        alpaka::wait::wait( queue );

        // Check device result against host result.
        Data arg;
        Data stdRes;
        TestStruct<
            TAcc,
            Data
        > t;
        for( Idx i( 0u ); i < numOps; ++i )
        {
            t = arr[i];
            switch( t.range )
            {
                case Range::POSITIVE_ONLY:
                    for( Idx j( 0 ); j < sizeArgs / 2 - 2; ++j )
                    {
                        arg = pBufHostArgs[j];
                        stdRes = t.stdOp( arg );
                        REQUIRE( stdRes ==
                                 Approx( pBufHostRes[j + i * sizeArgs] ) );
                    }
                    break;

                case Range::POSITIVE_AND_ZERO:
                    for( Idx j( 0 ); j < sizeArgs / 2 - 2; ++j )
                    {
                        arg = pBufHostArgs[j];
                        stdRes = t.stdOp( arg );
                        REQUIRE( stdRes ==
                                 Approx( pBufHostRes[j + i * sizeArgs] ) );
                    }
                    break;

                case Range::UNRESTRICTED:
                    for( Idx j( 0 ); j < sizeArgs / 2 - 2; ++j )
                    {
                        arg = pBufHostArgs[j];
                        stdRes = t.stdOp( arg );
                        REQUIRE( stdRes ==
                                 Approx( pBufHostRes[j + i * sizeArgs] ) );
                    }
                    break;

                case Range::ONE_NEIGHBOURHOOD:
                    if( sizeArgs < 4 )
                        break;
                    arg = 1;
                    stdRes = t.stdOp( arg );
                    REQUIRE( stdRes ==
                             Approx( pBufHostRes[0 + i * sizeArgs] ) );
                    arg = 0.5;
                    stdRes = t.stdOp( arg );
                    REQUIRE( stdRes ==
                             Approx( pBufHostRes[1 + i * sizeArgs] ) );
                    arg = 0;
                    stdRes = t.stdOp( arg );
                    REQUIRE( stdRes ==
                             Approx( pBufHostRes[2 + i * sizeArgs] ) );
                    arg = -0.5;
                    stdRes = t.stdOp( arg );
                    REQUIRE( stdRes ==
                             Approx( pBufHostRes[3 + i * sizeArgs] ) );
                    arg = -1;
                    stdRes = t.stdOp( arg );
                    REQUIRE( stdRes ==
                             Approx( pBufHostRes[4 + i * sizeArgs] ) );
                    break;

                default:
                    break;
            }
        }
    }
};

TEST_CASE("unaryOps", "[unaryOps]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt< 1u >,
        std::size_t
    >;
    alpaka::meta::forEachType< TestAccs >( TestTemplate<double>( ) );
    alpaka::meta::forEachType< TestAccs >( TestTemplate<float>( ) );
}

