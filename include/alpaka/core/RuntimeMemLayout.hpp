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

#pragma once

#include <alpaka/traits/Dim.hpp>  // traits::dim::getDim
#include <alpaka/traits/Memory.hpp>     // traits::memory::getPitchBytes, ...

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

namespace alpaka
{
    namespace memory
    {
        namespace layout
        {
            //#############################################################################
            //! The runtime memory layout interface.
            //!
            //! This defines the pitches of the memory buffer.
            //#############################################################################
            template<
                typename TDim>
            struct RuntimeMemLayout;

            //#############################################################################
            //! The 1D runtime memory layout.
            //#############################################################################
            template<>
            struct RuntimeMemLayout<
                dim::Dim1>
            {};
            //#############################################################################
            //! The 2D runtime memory layout.
            //#############################################################################
            template<>
            struct RuntimeMemLayout<
                dim::Dim2>
            {
                std::size_t uiRowPitchBytes;    //!< The width in bytes of the 2D array pointed to, including any padding added to the end of each row.
            };
            //#############################################################################
            //! The 3D runtime memory layout.
            //#############################################################################
            template<>
            struct RuntimeMemLayout<
                dim::Dim3>
            {
                std::size_t uiRowPitchBytes;    //!< The width in bytes of the 3D array pointed to, including any padding added to the end of each row.
                std::size_t uiRowWidthBytes;    //!< The width of each row in bytes.
                std::size_t uiSliceHeightRows;  //!< The height of each 2D slice in rows.
            };
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The RuntimeMemLayout dimension get specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetDim<
                alpaka::memory::layout::RuntimeMemLayout<TDim>>
            {
                using type = TDim;
            };
        }
    }
}