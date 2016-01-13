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

#include <iostream>                                 // std::cout, std::endl
#include <array>                                    // std::array
#include <numeric>                                  // std::iota 


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
            std::cout << "                              alpaka copyView test                              " << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << std::endl;

            /***************************************************************************
             * Configure types
             **************************************************************************/
            using Dim     = alpaka::dim::DimInt<3>;
            using Size    = std::size_t;
            using Extents = Size;
            using Host    = alpaka::acc::AccCpuSerial<Dim, Size>;
            using Acc     = alpaka::acc::AccCpuSerial<Dim, Size>;
            using DevHost = alpaka::dev::Dev<Host>;
            using DevAcc  = alpaka::dev::Dev<Acc>;
            using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
            using Stream  = alpaka::stream::StreamCpuSync;

            /***************************************************************************
             * Get the first device
             **************************************************************************/
            DevAcc  devAcc  (alpaka::dev::DevMan<Acc>::getDevByIdx(0));
            DevHost devHost (alpaka::dev::DevMan<Acc>::getDevByIdx(0));

            /***************************************************************************
             * Create sync stream
             **************************************************************************/
            Stream  stream  (devAcc);

            /***************************************************************************
             * Create buffers
             **************************************************************************/
            using Data = std::uint32_t;
            const Extents nElementsPerDim = 2;
            const Extents nElementsPerDimView = 1;            
            
            const alpaka::Vec<Dim, Size> extents(static_cast<Size>(nElementsPerDim),
                                                 static_cast<Size>(nElementsPerDim),
                                                 static_cast<Size>(nElementsPerDim));

            const alpaka::Vec<Dim, Size> extentsView(static_cast<Size>(nElementsPerDimView),
                                                     static_cast<Size>(nElementsPerDimView),
                                                     static_cast<Size>(nElementsPerDimView));

            const alpaka::Vec<Dim, Size> offsetView(static_cast<Size>(nElementsPerDimView),
                                                    static_cast<Size>(nElementsPerDimView),
                                                    static_cast<Size>(nElementsPerDimView));

            
            using ViewPlainPtr = alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size>;
            using Buf          = alpaka::mem::buf::Buf<DevHost, Data, Dim, Size>;
            
            std::array<Data, Dim::value * nElementsPerDim> plainBuffer;
            std::iota
            ViewPlainPtr srcBuffer(plainBuffer.data(), devHost, extents);
            Buf          destBuffer(alpaka::mem::buf::alloc<Data, Size>(devHost, extentsView));

            
            /***************************************************************************
             * Create view
             **************************************************************************/
            using ViewSubView  = alpaka::mem::view::ViewSubView<DevHost, Data, Dim, Size>;
            ViewSubView  viewBuffer(srcBuffer, extentsView, offsetView);

            
            /***************************************************************************
             * Copy view to destination buffer
             **************************************************************************/
            alpaka::mem::view::copy(stream, viewBuffer, destBuffer, extentsView);


            /***************************************************************************
             * Test results
             **************************************************************************/
            
            
            

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
