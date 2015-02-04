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

#include <alpaka/alpaka.hpp>                // alpaka::createKernelExecutor<...>

#include <chrono>                           // std::chrono::high_resolution_clock
#include <cassert>                          // assert
#include <iostream>                         // std::cout
#include <typeinfo>                         // typeid
#include <utility>                          // std::forward
#include <fstream>                          // std::ofstream

#include <boost/mpl/for_each.hpp>           // boost::mpl::for_each


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
    ALPAKA_FCT_HOST_ACC SimpleComplex(
        T a, 
        T b ) : 
            r(a), 
            i(b) 
    {}
    //-----------------------------------------------------------------------------
    //! 
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_HOST_ACC T abs()
    {
        return std::sqrt(r*r + i*i);
    }
    //-----------------------------------------------------------------------------
    //! 
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_HOST_ACC SimpleComplex operator*(SimpleComplex const & a)
    {
        return SimpleComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    //-----------------------------------------------------------------------------
    //! 
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_HOST_ACC SimpleComplex operator*(float const & a)
    {
        return SimpleComplex(r*a, i*a);
    }
    //-----------------------------------------------------------------------------
    //! 
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_HOST_ACC SimpleComplex operator+(SimpleComplex const & a)
    {
        return SimpleComplex(r+a.r, i+a.i);
    }
    //-----------------------------------------------------------------------------
    //! 
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_HOST_ACC SimpleComplex operator+(float const & a)
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
template<
    typename TAcc = alpaka::IAcc<>>
class MandelbrotKernel
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_ACC MandelbrotKernel()
    {
#ifndef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
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
#endif
    }

    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param pColors The output image.
    //! \param uiNumRows The number of rows in the image
    //! \param uiNumCols The number of columns in the image.
    //! \param uiPitchElems The pitch size in elements in the image.
    //! \param uiMaxIterations The maximum number of iterations.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_ACC void operator()(
        TAcc const & acc,
        std::uint32_t * const pColors,
        std::uint32_t const & uiNumRows, 
        std::uint32_t const & uiNumCols,
        std::uint32_t const & uiPitchElems,
        float const & fMinR,
        float const & fMaxR,
        float const & fMinI,
        float const & fMaxI,
        std::uint32_t const & uiMaxIterations) const
    {
        auto const uiGridKernelIdxX(acc.template getIdx<alpaka::Grid, alpaka::Kernels>()[0u]);
        auto const uiGridKernelIdxY(acc.template getIdx<alpaka::Grid, alpaka::Kernels>()[1u]);

        if((uiGridKernelIdxY < uiNumRows) && (uiGridKernelIdxX < uiNumCols))
        {
            SimpleComplex<float> c(
                (fMinR + (static_cast<float>(uiGridKernelIdxX)/float(uiNumCols-1)*(fMaxR - fMinR))) ,
                (fMinI + (static_cast<float>(uiGridKernelIdxY)/float(uiNumRows-1)*(fMaxI - fMinI))));

            auto const uiIterationCount(iterateMandelbrot(c, uiMaxIterations));

            pColors[uiGridKernelIdxY * uiPitchElems + uiGridKernelIdxX] = 
#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
                iterationCountToContinousColor(uiIterationCount, uiMaxIterations);
#else
                iterationCountToRepeatedColor(uiIterationCount);
#endif
        }
    }
    //-----------------------------------------------------------------------------
    //! \return 
    //!     The number of iterations until the Mandelbrot iteration with the given Value reaches the absolute value of 2.
    //!     Only does uiMaxIterations steps and returns uiMaxIterations if the value would be higher.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_ACC static std::uint32_t iterateMandelbrot(
        SimpleComplex<float> const & c,
        std::uint32_t const & uiMaxIterations)
    {
        SimpleComplex<float> z(0.0f, 0.0f);
        std::uint32_t iterations(0);
        for(; iterations<uiMaxIterations; ++iterations)
        {
            z = z*z + c;
            if(z.abs() > 2.0f)
            {
                break;
            }
        }
        return iterations;
    }
    
    //-----------------------------------------------------------------------------
    //! 
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_ACC static std::uint32_t convertRgbSingleToBgra(
        std::uint32_t const & r,
        std::uint32_t const & g,
        std::uint32_t const & b)
    {
        return 0xFF000000 | (r<<16) | (g<<8) | b;
    }

#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    //-----------------------------------------------------------------------------
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_ACC static std::uint32_t iterationCountToContinousColor(
        std::uint32_t const & uiIterationCount,
        std::uint32_t const & uiMaxIterations)
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
    ALPAKA_FCT_ACC std::uint32_t iterationCountToRepeatedColor(
        std::uint32_t const & uiIterationCount) const
    {
        return m_aColors[uiIterationCount%16];
    }

    ALPAKA_ALIGN(std::uint32_t, m_aColors[16]);
#endif
};

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<
    typename TExec,
    typename TStream,
    typename... TArgs>
void profileAcceleratedKernel(
    TExec const & exec,
    TStream const & stream, // \TODO: Add a getStream Method to the kernel executor and do not require this parameter!
    TArgs && ... args)
{
    std::cout
        << "profileAcceleratedKernel("
        << " kernelExecutor: " << typeid(TExec).name()
        << ")" << std::endl;

    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the accelerated kernel.
    exec(std::forward<TArgs>(args)...);
    
    // Wait for the stream to finish the kernel execution to measure its run time.
    alpaka::wait::wait(stream);

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//-----------------------------------------------------------------------------
//! Profiles the Mandelbrot kernel and checks the result.
//-----------------------------------------------------------------------------
struct MandelbrotKernelTester
{
    template<
        typename TAcc>
    void operator()(
        TAcc,
        std::size_t const & uiNumRows,
        std::size_t const & uiNumCols,
        float const & fMinR,
        float const & fMaxR,
        float const & fMinI,
        float const & fMaxI,
        std::size_t const & uiMaxIterations)
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Kernel = MandelbrotKernel<>;
        
        alpaka::Vec<2u> const v2uiExtents(
            static_cast<alpaka::Vec<2u>::Value>(uiNumCols),
            static_cast<alpaka::Vec<2u>::Value>(uiNumRows)
        );

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::Vec<3u> v3uiGridKernels(static_cast<alpaka::Vec<3u>::Value>(uiNumCols), static_cast<alpaka::Vec<3u>::Value>(uiNumRows), static_cast<alpaka::Vec<3u>::Value>(1u));
        alpaka::workdiv::BasicWorkDiv const workDiv(alpaka::workdiv::getValidWorkDiv<boost::mpl::vector<TAcc>>(v3uiGridKernels, false));

        std::cout
            << "profileAcceleratedMatMulKernel("
            << " uiNumRows:" << uiNumRows
            << ", uiNumCols:" << uiNumCols
            << ", uiMaxIterations:" << uiMaxIterations
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(Kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // allocate host memory
        std::vector<std::uint32_t> vColHost(uiNumRows * uiNumCols, 0u);
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::mem::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::mem::alloc<Type, MemSpaceHost>, which is not used here.
        // Instead we use MemBufPlainPtrWrapper to wrap the data.
        using MemBufWrapper = alpaka::mem::MemBufPlainPtrWrapper<
            alpaka::mem::MemSpaceHost,
            std::uint32_t,
            alpaka::dim::Dim2>;
        MemBufWrapper memBufColHost(vColHost.data(), v2uiExtents);
        
        // Allocate the buffer on the accelerator.
        using AccMemSpace = typename alpaka::mem::GetMemSpaceT<TAcc>;
        auto memBufColAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(v2uiExtents));
        
        // Get a new stream.
        alpaka::stream::GetStreamT<TAcc> stream;

        // Copy Host -> Acc.
        alpaka::mem::copy(memBufColAcc, memBufColHost, v2uiExtents, stream);
        
        // Build the kernel executor.
        auto exec(alpaka::createKernelExecutor<TAcc, Kernel>());
        // Profile the kernel execution.
        profileAcceleratedKernel(exec(workDiv, stream),
            stream,
            alpaka::mem::getNativePtr(memBufColAcc),
            static_cast<std::uint32_t>(uiNumRows),
            static_cast<std::uint32_t>(uiNumCols),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufColAcc)),
            fMinR,
            fMaxR,
            fMinI,
            fMaxI,
            static_cast<std::uint32_t>(uiMaxIterations));

        // Copy back the result.
        alpaka::mem::copy(memBufColHost, memBufColAcc, v2uiExtents, stream);
        
        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);
        
        std::string const sFileName("mandelbrot"+std::to_string(uiNumCols)+"x"+std::to_string(uiNumRows)+"_"+alpaka::acc::getAccName<TAcc>()+".tga");
        std::ofstream ofs(
            sFileName, 
            std::ofstream::out | std::ofstream::binary);
        if (!ofs.is_open())
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
        ofs.put((uiNumCols & 0xFF));        // Width of Image.
        ofs.put((uiNumCols >> 8) & 0xFF);
        ofs.put((uiNumRows & 0xFF));        // Height of Image.
        ofs.put((uiNumRows >> 8) & 0xFF);
        ofs.put(0x20);                      // Image Pixel Size.
        ofs.put(0x20);                      // Image Descriptor Byte.
        // Write data.
        ofs.write(reinterpret_cast<char*>(vColHost.data()), vColHost.size()*sizeof(std::uint32_t));

        std::cout << "################################################################################" << std::endl;
    }
};

//-----------------------------------------------------------------------------
//! Program entry point.
//-----------------------------------------------------------------------------
int main()
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka mandelbrot test                              " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::acc::writeEnabledAccelerators(std::cout);

        std::cout << std::endl;

#ifdef ALPAKA_CUDA_ENABLED
        // Select the first CUDA device.
        // NOTE: This is not required to run any kernels on the CUDA accelerator because all accelerators have a default device. This only shows the possibility.
        alpaka::dev::GetDevManT<alpaka::AccCuda>::setCurrentDevice(
            alpaka::dev::GetDevManT<alpaka::AccCuda>::getCurrentDevice());
#endif
        MandelbrotKernelTester mandelbrotTester;

        // For different sizes.
        for(std::size_t uiSize(1u<<8);
#if ALPAKA_INTEGRATION_TEST
            uiSize <= 1u<<8;
#else
            uiSize <= 1u<<13;
#endif
            uiSize *= 2u)
        {
            std::cout << std::endl;

            // Execute the kernel on all enabled accelerators.
            boost::mpl::for_each<alpaka::acc::EnabledAccelerators>(
                std::bind(
                    mandelbrotTester,
                    std::placeholders::_1,
                    uiSize,
                    uiSize,
                    -2.0f,
                    +1.0f,
                    -1.2f,
                    +1.2f,
                    300u));
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
