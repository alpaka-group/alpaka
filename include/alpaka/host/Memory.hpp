/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/host/MemorySpace.hpp>  // MemorySpaceHost

#include <alpaka/interfaces/Memory.hpp> // alpaka::MemCopy

#include <cstring>                      // std::memcpy, std::memset
#include <cassert>                      // assert
#include <cstdint>                      // std::uint8_t

namespace alpaka
{
    namespace memory
    {
        namespace detail
        {
            //#############################################################################
            //! Allocates memory in the Host memory space.
            //#############################################################################
            template<>
            struct MemAlloc<MemorySpaceHost>
            {
                MemAlloc(void ** const pBuffer, std::size_t const uiSizeBytes)
                {
                    assert(uiSizeBytes>0);

                    (*pBuffer) = reinterpret_cast<void *>(new std::uint8_t[uiSizeBytes]);
                    assert((*pBuffer));
                }
            };

            //#############################################################################
            //! Frees memory in the Host memory space.
            //#############################################################################
            template<>
            struct MemFree<MemorySpaceHost>
            {
                MemFree(void * const pBuffer)
                {
                    assert(pBuffer);
                    delete[] pBuffer;
                }
            };
            //#############################################################################
            //! Copies from a host memory into host memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceHost, MemorySpaceHost>
            {
                MemCopy(void * const pBufferDst, void * const pBufferSrc, size_t const uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    std::memcpy(pBufferDst, pBufferSrc, uiSizeBytes);
                }
            };

            //#############################################################################
            //! Fills the buffer with the data in the Host memory space.
            //#############################################################################
            template<>
            struct MemSet<MemorySpaceHost>
            {
                MemSet(void * const pBuffer, int const iValue, size_t const uiSizeBytes)
                {
                    assert(pBuffer);

                    std::memset(pBuffer, iValue, uiSizeBytes);
                }
            };
        }
    }
}
