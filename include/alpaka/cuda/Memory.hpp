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

#include <alpaka/cuda/MemorySpace.hpp>  // MemorySpaceCuda

#include <alpaka/host/MemorySpace.hpp>  // MemorySpaceHost

#include <alpaka/interfaces/Memory.hpp> // alpaka::MemCopy

#include <alpaka/cuda/Common.hpp>

#include <cstddef>                      // std::size_t
#include <cassert>                      // assert

namespace alpaka
{
    namespace memory
    {
        namespace detail
        {
            //#############################################################################
            //! Allocates memory in the CUDA memory space.
            //#############################################################################
            template<>
            struct MemAlloc<MemorySpaceCuda>
            {
                MemAlloc(void ** const pBuffer, std::size_t const uiSizeBytes)
                {
                    assert(uiSizeBytes>0);

                    ALPAKA_CUDA_CHECK(cudaMalloc(pBuffer, uiSizeBytes));
                    assert((*pBuffer));
                }
            };

            //#############################################################################
            //! Allocates memory in the CUDA memory space.
            //#############################################################################
            template<>
            struct MemFree<MemorySpaceCuda>
            {
                MemFree(void * pBuffer)
                {
                    assert(pBuffer);
                    ALPAKA_CUDA_CHECK(cudaFree(pBuffer));
                }
            };

            //#############################################################################
            //! Copies from CUDA memory into Host memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceHost, MemorySpaceCuda>
            {
                MemCopy(void * const pBufferDst, void const * const pBufferSrc, size_t const uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    ALPAKA_CUDA_CHECK(cudaMemcpy(pBufferDst, pBufferSrc, uiSizeBytes, cudaMemcpyDeviceToHost));
                }
            };
            //#############################################################################
            //! Copies from Host memory into CUDA memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceCuda, MemorySpaceHost>
            {
                MemCopy(void * const pBufferDst, void const * const pBufferSrc, size_t const uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    ALPAKA_CUDA_CHECK(cudaMemcpy(pBufferDst, pBufferSrc, uiSizeBytes, cudaMemcpyHostToDevice));
                }
            };
            //#############################################################################
            //! Copies from CUDA memory into CUDA memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceCuda, MemorySpaceCuda>
            {
                MemCopy(void * const pBufferDst, void const * const pBufferSrc, size_t const uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    ALPAKA_CUDA_CHECK(cudaMemcpy(pBufferDst, pBufferSrc, uiSizeBytes, cudaMemcpyDeviceToDevice));
                }
            };

            //#############################################################################
            //! Fills the buffer with the data in the CUDA memory space.
            //#############################################################################
            template<>
            struct MemSet<MemorySpaceCuda>
            {
                MemSet(void * const pBuffer, int const iValue, size_t const uiSizeBytes)
                {
                    assert(pBuffer);

                    ALPAKA_CUDA_CHECK(cudaMemset(pBuffer, iValue, uiSizeBytes));
                }
            };
        }
    }
}
