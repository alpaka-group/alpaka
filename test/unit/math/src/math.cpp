/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "Defines.hpp"
#include "Buffer.hpp"
#include "Functor.hpp"
#include "DataGen.hpp"

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <catch2/catch.hpp>

struct TestKernel
{
    //! @tparam TAcc Accelerator.
    //! @tparam TFunctor Functor defined in Functor.hpp.
    //! @param acc Accelerator given from alpaka.
    //! @param functor Accessible with operator().
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc,
             typename TResults,
             typename TFunctor,
             typename TArgs>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TResults const & results,
        TFunctor const & functor,
        TArgs const & args) const noexcept
        -> void
    {
        for( size_t i = 0; i < TArgs::capacity; ++i )
        {
          results(i, acc) = functor(args(i, acc), acc);
        }
    }
};


//#############################################################################
// The TestTemplate runs the main code and the tests (Buffer,Functor,...).
//! @tparam TAcc One of the possible accelerator types, that need to be tested.
//! @tparam TData By now either double or float.
template <
    typename TAcc,
    typename TData>
struct TestTemplate
{
    template < typename TFunctor >
    auto operator() ( unsigned long seed ) -> void
    {
        // SETUP (defines and initialising)
        // DevAcc and DevHost are defined in Buffer.hpp too.
        using DevAcc = alpaka::dev::Dev< TAcc >;
        using DevHost = alpaka::dev::DevCpu;
        using PltfAcc = alpaka::pltf::Pltf< DevAcc >;
        using PltfHost = alpaka::pltf::Pltf< DevHost >;

        using Dim = alpaka::dim::DimInt< 1u >;
        using Idx = std::size_t;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
        using QueueAcc = alpaka::test::queue::DefaultQueue< DevAcc >;
        using TArgsItem = alpaka::test::unit::math::ArgsItem<TData, TFunctor::arity>;

        static constexpr auto capacity = 1000;

        using Args = alpaka::test::unit::math::Buffer<
            TAcc,
            TArgsItem,
            capacity
        >;
        using Results = alpaka::test::unit::math::Buffer<
            TAcc,
            TData,
            capacity
        >;

        // Every functor is executed individual on one kernel.
        static constexpr size_t elementsPerThread = 1u;
        static constexpr size_t sizeExtent = 1u;

        DevAcc const devAcc{ alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) };
        DevHost const devHost{ alpaka::pltf::getDevByIdx< PltfHost >( 0u ) };

        QueueAcc queue{ devAcc };

        TestKernel kernel;
        TFunctor functor;
        Args args{ devAcc };
        Results results{ devAcc };

        WorkDiv const workDiv{
            alpaka::workdiv::getValidWorkDiv< TAcc >(
                devAcc,
                sizeExtent,
                elementsPerThread,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
            )};
        // SETUP COMPLETED.

        // Fill the buffer with random test-numbers.
        alpaka::test::unit::math::fillWithRndArgs<TData>( args, functor, seed );
        for( size_t i = 0; i < Results::capacity; ++i )
            results(i) = static_cast<TData>(std::nan( "" ));

        // Copy both buffer to the device
        args.copyToDevice(queue);
        results.copyToDevice(queue);

        auto const taskKernel(
            alpaka::kernel::createTaskKernel< TAcc >(
                workDiv,
                kernel,
                results,
                functor,
                args
            )
        );
        // Enqueue the kernel execution task.
        alpaka::queue::enqueue( queue, taskKernel );
        // Copy back the results (encapsulated in the buffer class).
        results.copyFromDevice( queue );
        alpaka::wait::wait( queue );
        std::cout.precision( std::numeric_limits<TData>::digits10 + 1 );

        INFO("Operator: " << functor)
        INFO("Type: " << typeid( TData ).name() ) // Compiler specific.
#if ALPAKA_DEBUG_FULL
        INFO("The args buffer: \n" << std::setprecision(
            std::numeric_limits<TData>::digits10 + 1)
            << args << "\n")
#endif
        for( size_t i = 0; i < Args::capacity; ++i )
        {
            INFO("Idx i: " << i)
            TData std_result = functor(args(i));
            REQUIRE( results(i) == Approx(std_result) );
        }
    }
};

template< typename TData >
struct ForEachFunctor
{
    template< typename TAcc >
    auto operator()( unsigned long seed ) -> void
    {
        alpaka::meta::forEachType < alpaka::test::unit::math::UnaryFunctors >(
            TestTemplate<
                TAcc,
                TData
            >( ),
            seed
        );

        alpaka::meta::forEachType< alpaka::test::unit::math::BinaryFunctors >(
            TestTemplate<
                TAcc,
                TData
            >( ),
            seed
        );
    }
};

TEST_CASE("mathOps", "[math] [operator]")
{
    /*
     * All alpaka::math:: functions are tested here except sincos.
     * The function will be called with a buffer from the custom Buffer class.
     * This argument Buffer contains ArgsItems from Defines.hpp and can be
     * accessed with the overloaded operator().
     * The args Buffer looks similar like [[0, 1], [2, 3], [4, 5]],
     * where every sub-list makes one functor-call so the result Buffer would be:
     * [f(0, 1), f(2, 3), f(4, 5)].
     * The results are saved in a different Buffer witch contains plain data.
     * The results are than compared to the result of a std:: implementation.
     * The default result is nan and should fail a test.
     *
     * BE AWARE that:
     * - ALPAKA_CUDA_FAST_MATH should be disabled
     * - not all casts between float and double can be detected.
     * - no explicit edge cases are tested, rather than 0, maximum and minimum
     *   - but it is easy to add a new Range:: enum-type with custom edge cases
     *  - some tests may fail if ALPAKA_CUDA_FAST_MATH is turned on
     * - nan typically fails every test, but could be normal defined behaviour
     * - inf/-inf typically dont fail a test
     * - for easy debugging the << operator is overloaded for Buffer objects
     * - arguments are generated between 0 and 1000
     *     and the default argument-buffer-extent is 1000
     * The arguments are generated in DataGen.hpp and can easily be modified.
     * The arguments depend on the Range:: enum-type specified for each functor.
     * ----------------------------------------------------------------------
     * TEST_CASE        | sets seed          | specifies datatype & acc-list
     * ForEachFunctor   |       -            | specifies functor
     *                                         (from Functor.hpp)
     * TestTemplate     | functor, device, host, queue, kernel, usw.
     * - main execution:
     * - each functor has an arity and a array of ranges
     *     - there is one args Buffer and one results Buffer
     *         - each buffer encapsulated the host/device communication
     *         - as well as the data access and the initialisation
     * - all operators are tested independent, one per kernel
     * - tests the results against the std implementation ( catch REQUIRES)
     *
     * TestKernel
     * - uses the alpaka::math:: option from the functor
     * - uses the device-buffer  option from the args
     *
     * EXTENSIBILITY:
     * - Add new operators in Functor.hpp and add them to the ...Functors tuple.
     * - Add a new Range:: enum-type in Defines.hpp
     *     - specify a fill-method in DataGen.hpp
     * - Add a new Arity:: enum-type in Defines.hpp
     *     - add a matching operator() function in Functor.hpp,
     *     - add a new ...Functors tuple
     *     - call alpaka::meta::forEachType with the tuple in ForEachFunctor
     */

    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt< 1u >,
        std::size_t
    >;
    const unsigned long seed = 1337;
    std::cout << "using seed: " << seed << "\n\n";
    std::cout << "testing:\n "
        << std::tuple_size<TestAccs>::value
        << " - accelerators !\n"
        << std::tuple_size<alpaka::test::unit::math::UnaryFunctors>::value
        << " - unary math operators\n"
        << std::tuple_size<alpaka::test::unit::math::BinaryFunctors>::value
        << " - binary math operators\n"
        << "testing with two data types\n"
        << "total 2 * accelerators * (unary + binary) * capacity\n\n";


    alpaka::meta::forEachType< TestAccs >(
        ForEachFunctor< double >( ),
        seed
    );

    alpaka::meta::forEachType< TestAccs >(
        ForEachFunctor< float >( ),
        seed
    );
}
