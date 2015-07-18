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
#include <fstream>                                  // std::ofstream
#include <algorithm>                                // std::replace

//#define ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING  // Define this to enable the continuous color mapping.

//#############################################################################
//! Complex Number.
//#############################################################################
template<
    typename T>
class SimpleComplex
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC SimpleComplex(
        T const & a,
        T const & b) :
            r(a),
            i(b)
    {}
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto absSq()
    -> T
    {
        return r*r + i*i;
    }
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator*(SimpleComplex const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator*(float const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r*a, i*a);
    }
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator+(SimpleComplex const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r+a.r, i+a.i);
    }
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator+(float const & a)
    -> SimpleComplex
    {
        return SimpleComplex(r+a, i);
    }

public:
    T r;
    T i;
};

//#############################################################################
//! A Mandelbrot kernel.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
class MandelbrotKernel
{
public:
#ifndef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    //-----------------------------------------------------------------------------
    //! Constructor.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC MandelbrotKernel()
    {
        // Banding can be prevented by a continuous color functions.
        m_aColors[0u] = convertRgbSingleToBgra(66, 30, 15);
        m_aColors[1u] = convertRgbSingleToBgra(25, 7, 26);
        m_aColors[2u] = convertRgbSingleToBgra(9, 1, 47);
        m_aColors[3u] = convertRgbSingleToBgra(4, 4, 73);
        m_aColors[4u] = convertRgbSingleToBgra(0, 7, 100);
        m_aColors[5u] = convertRgbSingleToBgra(12, 44, 138);
        m_aColors[6u] = convertRgbSingleToBgra(24, 82, 177);
        m_aColors[7u] = convertRgbSingleToBgra(57, 125, 209);
        m_aColors[8u] = convertRgbSingleToBgra(134, 181, 229);
        m_aColors[9u] = convertRgbSingleToBgra(211, 236, 248);
        m_aColors[10u] = convertRgbSingleToBgra(241, 233, 191);
        m_aColors[11u] = convertRgbSingleToBgra(248, 201, 95);
        m_aColors[12u] = convertRgbSingleToBgra(255, 170, 0);
        m_aColors[13u] = convertRgbSingleToBgra(204, 128, 0);
        m_aColors[14u] = convertRgbSingleToBgra(153, 87, 0);
        m_aColors[15u] = convertRgbSingleToBgra(106, 52, 3);
    }
#endif

    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \param acc The accelerator to be executed on.
    //! \param pColors The output image.
    //! \param uiNumRows The number of rows in the image
    //! \param uiNumCols The number of columns in the image.
    //! \param uiPitchElems The pitch size in elements in the image.
    //! \param fMinR The left border.
    //! \param fMaxR The right border.
    //! \param fMinI The bottom border.
    //! \param fMaxI The top border.
    //! \param uiMaxIterations The maximum number of iterations.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        std::uint32_t * const pColors,
        std::uint32_t const & uiNumRows,
        std::uint32_t const & uiNumCols,
        std::uint32_t const & uiPitchBytes,
        float const & fMinR,
        float const & fMaxR,
        float const & fMinI,
        float const & fMaxI,
        std::uint32_t const & uiMaxIterations) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 2,
            "The MandelbrotKernel expects 2-dimensional indices!");

        auto const uiGridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        auto const & uiGridThreadIdxX(uiGridThreadIdx[1u]);
        auto const & uiGridThreadIdxY(uiGridThreadIdx[0u]);

        if((uiGridThreadIdxY < uiNumRows) && (uiGridThreadIdxX < uiNumCols))
        {
            SimpleComplex<float> c(
                (fMinR + (static_cast<float>(uiGridThreadIdxX)/float(uiNumCols-1)*(fMaxR - fMinR))),
                (fMinI + (static_cast<float>(uiGridThreadIdxY)/float(uiNumRows-1)*(fMaxI - fMinI))));

            auto const uiIterationCount(iterateMandelbrot(c, uiMaxIterations));

            auto const pColorsRow(reinterpret_cast<std::uint32_t *>(reinterpret_cast<std::uint8_t *>(pColors) + uiGridThreadIdxY * uiPitchBytes));
            pColorsRow[uiGridThreadIdxX] =
#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
                iterationCountToContinousColor(uiIterationCount, uiMaxIterations);
#else
                iterationCountToRepeatedColor(uiIterationCount);
#endif
        }
    }
    //-----------------------------------------------------------------------------
    //! \return The number of iterations until the Mandelbrot iteration with the given Value reaches the absolute value of 2.
    //!     Only does uiMaxIterations steps and returns uiMaxIterations if the value would be higher.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC static auto iterateMandelbrot(
        SimpleComplex<float> const & c,
        std::uint32_t const & uiMaxIterations)
    -> std::uint32_t
    {
        SimpleComplex<float> z(0.0f, 0.0f);
        for(std::uint32_t iterations(0); iterations<uiMaxIterations; ++iterations)
        {
            z = z*z + c;
            if(z.absSq() > 4.0f)
            {
                return iterations;
            }
        }
        return uiMaxIterations;
    }

    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC static auto convertRgbSingleToBgra(
        std::uint32_t const & r,
        std::uint32_t const & g,
        std::uint32_t const & b)
    -> std::uint32_t
    {
        return 0xFF000000 | (r<<16) | (g<<8) | b;
    }

#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    //-----------------------------------------------------------------------------
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC static auto iterationCountToContinousColor(
        std::uint32_t const & uiIterationCount,
        std::uint32_t const & uiMaxIterations)
    -> std::uint32_t
    {
        // Map the iteration count on the 0..1 interval.
        float const t(static_cast<float>(uiIterationCount)/static_cast<float>(uiMaxIterations));
        float const oneMinusT(1.0f-t);
        // Use some modified Bernstein polynomials for r, g, b.
        std::uint32_t const r(static_cast<std::uint32_t>(9.0f*oneMinusT*t*t*t*255.0f));
        std::uint32_t const g(static_cast<std::uint32_t>(15.0f*oneMinusT*oneMinusT*t*t*255.0f));
        std::uint32_t const b(static_cast<std::uint32_t>(8.5f*oneMinusT*oneMinusT*oneMinusT*t*255.0f));
        return convertRgbSingleToBgra(r, g, b);
    }
#else
    //-----------------------------------------------------------------------------
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto iterationCountToRepeatedColor(
        std::uint32_t const & uiIterationCount) const
    -> std::uint32_t
    {
        return m_aColors[uiIterationCount%16];
    }

    alignas(16) std::uint32_t m_aColors[16];
#endif
};

//-----------------------------------------------------------------------------
//! Writes the buffer color data to a file.
//-----------------------------------------------------------------------------
template<
    typename TBuf>
auto writeTgaColorImage(
    std::string const & sFileName,
    TBuf const & bufRgba)
-> void
{
    static_assert(
        alpaka::dim::Dim<TBuf>::value == 2,
        "The buffer has to be 2 dimensional!");
    static_assert(
        std::is_integral<alpaka::mem::view::Elem<TBuf>>::value,
        "The buffer element type has to be integral!");

    // The width of the input buffer is in input elements.
    auto const uiBufWidthElems(alpaka::extent::getWidth(bufRgba));
    auto const uiBufWidthBytes(uiBufWidthElems * sizeof(alpaka::mem::view::Elem<TBuf>));
    // The row width in bytes has to be dividable by 4 Bytes (RGBA).
    assert(uiBufWidthBytes % sizeof(std::uint32_t) == 0);
    // The number of colors in a row.
    auto const uiBufWidthColors(uiBufWidthBytes / sizeof(std::uint32_t));
    assert(uiBufWidthColors >= 1);
    auto const uiBufHeightColors(alpaka::extent::getHeight(bufRgba));
    assert(uiBufHeightColors >= 1);
    auto const uiBufPitchBytes(alpaka::mem::view::getPitchBytes<alpaka::dim::Dim<TBuf>::value - 1u>(bufRgba));
    assert(uiBufPitchBytes >= uiBufWidthBytes);

    std::ofstream ofs(
        sFileName,
        std::ofstream::out | std::ofstream::binary);
    if(!ofs.is_open())
    {
        throw std::invalid_argument("Unable to open file: "+sFileName);
    }

    // Write tga image header.
    ofs.put(0x00);                      // Number of Characters in Identification Field.
    ofs.put(0x00);                      // Color Map Type.
    ofs.put(0x02);                      // Image Type Code.
    ofs.put(0x00);                      // Color Map Origin.
    ofs.put(0x00);
    ofs.put(0x00);                      // Color Map Length.
    ofs.put(0x00);
    ofs.put(0x00);                      // Color Map Entry Size.
    ofs.put(0x00);                      // X Origin of Image.
    ofs.put(0x00);
    ofs.put(0x00);                      // Y Origin of Image.
    ofs.put(0x00);
    ofs.put((uiBufWidthColors & 0xFF)); // Width of Image.
    ofs.put((uiBufWidthColors >> 8) & 0xFF);
    ofs.put((uiBufHeightColors & 0xFF));// Height of Image.
    ofs.put((uiBufHeightColors >> 8) & 0xFF);
    ofs.put(0x20);                      // Image Pixel Size.
    ofs.put(0x20);                      // Image Descriptor Byte.

    // Write the data.
    char const * pData(reinterpret_cast<char const *>(alpaka::mem::view::getPtrNative(bufRgba)));
    // If there is no padding, we can directly write the whole buffer data ...
    if(uiBufPitchBytes == uiBufWidthBytes)
    {
        ofs.write(
            pData,
            uiBufWidthColors*uiBufHeightColors);
    }
    // ... else we have to write row by row.
    else
    {
        for(auto uiRow(decltype(uiBufHeightColors)(0)); uiRow<uiBufHeightColors; ++uiRow)
        {
            ofs.write(
                pData + uiBufPitchBytes*uiRow,
                uiBufWidthColors);
        }
    }
}

//#############################################################################
//! Profiles the Mandelbrot kernel.
//#############################################################################
struct MandelbrotKernelTester
{
    template<
        typename TAcc,
        typename TSize>
    auto operator()(
        TSize const & uiNumRows,
        TSize const & uiNumCols,
        float const & fMinR,
        float const & fMaxR,
        float const & fMinI,
        float const & fMaxI,
        TSize const & uiMaxIterations)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Val = std::uint32_t;

        // Create the kernel function object.
        MandelbrotKernel kernel;

        // Get the host device.
        auto devHost(alpaka::dev::cpu::getDev());

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::stream::Stream<alpaka::dev::Dev<TAcc>> stream(
            alpaka::stream::create(devAcc));

        alpaka::Vec2<TSize> const v2uiExtents(
            static_cast<TSize>(uiNumRows),
            static_cast<TSize>(uiNumCols));

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<2u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtents,
                false,
                alpaka::workdiv::GridBlockExtentsSubDivRestrictions::Unrestricted));

        std::cout
            << "MandelbrotKernelTester("
            << " uiNumRows:" << uiNumRows
            << ", uiNumCols:" << uiNumCols
            << ", uiMaxIterations:" << uiMaxIterations
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // allocate host memory
        auto bufColorHost(
            alpaka::mem::buf::alloc<Val, TSize>(devHost, v2uiExtents));

        // Allocate the buffer on the accelerator.
        auto bufColorAcc(
            alpaka::mem::buf::alloc<Val, TSize>(devAcc, v2uiExtents));

        // Copy Host -> Acc.
        alpaka::mem::view::copy(bufColorAcc, bufColorHost, v2uiExtents, stream);

        // Create the executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));
        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::examples::measureKernelRunTimeMs(
                exec,
                kernel,
                alpaka::mem::view::getPtrNative(bufColorAcc),
                uiNumRows,
                uiNumCols,
                alpaka::mem::view::getPitchBytes<1u>(bufColorAcc),
                fMinR,
                fMaxR,
                fMinI,
                fMaxI,
                uiMaxIterations)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(bufColorHost, bufColorAcc, v2uiExtents, stream);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Write the image to a file.
        std::string sFileName("mandelbrot"+std::to_string(uiNumCols)+"x"+std::to_string(uiNumRows)+"_"+alpaka::acc::getAccName<TAcc>()+".tga");
        std::replace(sFileName.begin(), sFileName.end(), '<', '_');
        std::replace(sFileName.begin(), sFileName.end(), '>', '_');
        writeTgaColorImage(
            sFileName,
            bufColorHost);

        std::cout << "################################################################################" << std::endl;
    }
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
        std::cout << "                            alpaka mandelbrot test                              " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::examples::accs::writeEnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>(std::cout);

        std::cout << std::endl;

        MandelbrotKernelTester mandelbrotTester;

        // For different sizes.
        for(std::uint32_t uiSize(1u<<3u);
#if ALPAKA_INTEGRATION_TEST
            uiSize <= 1u<<8u;
#else
            uiSize <= 1u<<13u;
#endif
            uiSize *= 2u)
        {
            std::cout << std::endl;

            // Execute the kernel on all enabled accelerators.
            alpaka::core::forEachType<
                alpaka::examples::accs::EnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>>(
                    mandelbrotTester,
                    uiSize,
                    uiSize,
                    -2.0f,
                    +1.0f,
                    -1.2f,
                    +1.2f,
                    300u);
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
