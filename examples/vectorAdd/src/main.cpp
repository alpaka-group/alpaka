/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>                        // alpaka::exec::create
#include <alpaka/examples/MeasureKernelRunTime.hpp> // measureKernelRunTimeMs
#include <alpaka/examples/accs/EnabledAccs.hpp>     // EnabledAccs

#include <chrono>                                   // std::chrono::high_resolution_clock
#include <cassert>                                  // assert
#include <iostream>                                 // std::cout
#include <typeinfo>                                 // typeid
#include <utility>                                  // std::forward

//#############################################################################
//! A vector addition kernel.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
class VectorAddKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TSize const & numElements) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        auto const gridThreadIdxX(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        if (gridThreadIdxX < numElements)
        {
            C[gridThreadIdxX] = A[gridThreadIdxX] + B[gridThreadIdxX];
        }
    }
};

//#############################################################################
//! Profiles the vector addition kernel.
//#############################################################################
struct VectorAddKernelTester
{
    template<
        typename TAcc,
        typename TSize>
    auto operator()(
        TSize const & numElements)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Val = float;

        // Create the kernel function object.
        VectorAddKernel kernel;

        // Get the host device.
        auto devHost(alpaka::dev::cpu::getDev());

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::examples::Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        alpaka::Vec1<TSize> const v1uiExtents(
            numElements);

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v1uiExtents,
                false,
                alpaka::workdiv::GridBlockExtentsSubDivRestrictions::Unrestricted));

        std::cout
            << "VectorAddKernelTester("
            << " numElements:" << numElements
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate host memory buffers.
        auto memBufHostA(alpaka::mem::buf::alloc<Val, TSize>(devHost, v1uiExtents));
        auto memBufHostB(alpaka::mem::buf::alloc<Val, TSize>(devHost, v1uiExtents));
        auto memBufHostC(alpaka::mem::buf::alloc<Val, TSize>(devHost, v1uiExtents));

        // Initialize the host input vectors
        for (TSize i(0); i < numElements; ++i)
        {
            alpaka::mem::view::getPtrNative(memBufHostA)[i] = static_cast<Val>(rand());
            alpaka::mem::view::getPtrNative(memBufHostB)[i] = static_cast<Val>(rand());
        }

        // Allocate the buffer on the accelerator.
        auto memBufAccA(alpaka::mem::buf::alloc<Val, TSize>(devAcc, v1uiExtents));
        auto memBufAccB(alpaka::mem::buf::alloc<Val, TSize>(devAcc, v1uiExtents));
        auto memBufAccC(alpaka::mem::buf::alloc<Val, TSize>(devAcc, v1uiExtents));

        // Copy Host -> Acc.
        alpaka::mem::view::copy(stream, memBufAccA, memBufHostA, v1uiExtents);
        alpaka::mem::view::copy(stream, memBufAccB, memBufHostB, v1uiExtents);

        // Create the executor task.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            alpaka::mem::view::getPtrNative(memBufAccA),
            alpaka::mem::view::getPtrNative(memBufAccB),
            alpaka::mem::view::getPtrNative(memBufAccC),
            numElements));

        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::examples::measureKernelRunTimeMs(
                stream,
                exec)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(stream, memBufHostC, memBufAccC, v1uiExtents);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        bool resultCorrect(true);
        auto const pHostData(alpaka::mem::view::getPtrNative(memBufHostC));
        for(TSize i(0u);
            i < numElements;
            ++i)
        {
            auto const & val(pHostData[i]);
            auto const correctResult(alpaka::mem::view::getPtrNative(memBufHostA)[i]+alpaka::mem::view::getPtrNative(memBufHostB)[i]);
            if(val != correctResult)
            {
                std::cout << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
                resultCorrect = false;
            }
        }

        if(resultCorrect)
        {
            std::cout << "Execution results correct!" << std::endl;
        }

        std::cout << "################################################################################" << std::endl;

        allResultsCorrect = allResultsCorrect && resultCorrect;
    }

public:
    bool allResultsCorrect = true;
};

//-----------------------------------------------------------------------------
//! Program entry point.
//-----------------------------------------------------------------------------
auto main()
-> int
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka vector add test                              " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::examples::accs::writeEnabledAccs<alpaka::dim::DimInt<1u>, std::size_t>(std::cout);

        std::cout << std::endl;

        VectorAddKernelTester vectorAddKernelTester;

        // For different sizes.
#if ALPAKA_INTEGRATION_TEST
        for(std::size_t vecSize(1u); vecSize <= 1u<<9u; vecSize *= 8u)
#else
        for(std::size_t vecSize(1u); vecSize <= 1u<<16u; vecSize *= 2u)
#endif
        {
            std::cout << std::endl;

            // Execute the kernel on all enabled accelerators.
            alpaka::core::forEachType<
                alpaka::examples::accs::EnabledAccs<alpaka::dim::DimInt<1u>, std::size_t>>(
                    vectorAddKernelTester,
                    vecSize);
        }
        return EXIT_SUCCESS;
    }
    catch(std::exception const & e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unknown Exception" << std::endl;
        return EXIT_FAILURE;
    }
}
