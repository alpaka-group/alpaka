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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC, ALPAKA_ALIGN

#include <cstdint>                  // std::size_t

namespace alpaka
{
    //#############################################################################
    //! A pitched pointer.
    //#############################################################################
    struct PichedPtr
    {
        std::size_t pitch;
        void * ptr;
        std::size_t xsize;
        std::size_t ysize;
    };

    PichedPtr makePitchedPtr(
        void * d,
        std::size_t p,
        std::size_t xsz,
        std::size_t ysz)
    {
        return PichedPtr{p, d, xsz, ysz};
    }

}