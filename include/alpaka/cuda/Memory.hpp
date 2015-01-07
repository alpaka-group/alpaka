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
            //! Page-locks the memory range specified.
            //#############################################################################
            void pageLockHostMem(void const * const pBuffer, std::size_t const uiSizeBytes)
            {
                assert(pBuffer);
                assert(uiSizeBytes>0);

                // cudaHostRegisterDefault: 
                //  See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                // cudaHostRegisterPortable: 
                //  The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
                // cudaHostRegisterMapped: 
                //  Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                //  This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                ALPAKA_CUDA_CHECK(cudaHostRegister(const_cast<void *>(pBuffer), uiSizeBytes, cudaHostRegisterDefault));
            }
            //#############################################################################
            //! Unmaps page-locked memory.
            //#############################################################################
            void unPageLockHostMem(void const * const pBuffer)
            {
                assert(pBuffer);

                ALPAKA_CUDA_CHECK(cudaHostUnregister(const_cast<void *>(pBuffer)));
            }

            //#############################################################################
            //! The CUDA memory allocator.
            //#############################################################################
            template<>
            struct MemAlloc<MemorySpaceCuda>
            {
                MemAlloc(void ** const pBuffer, std::size_t const & uiSizeBytes)
                {
                    assert(uiSizeBytes>0);

                    ALPAKA_CUDA_CHECK(cudaMalloc(pBuffer, uiSizeBytes));
                    assert((*pBuffer));
                }
            };

            //#############################################################################
            //! The CUDA memory freer.
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
            //! The CUDA memory copier.
            //!
            //! Copies from CUDA memory into Host memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceHost, MemorySpaceCuda>
            {
                MemCopy(void * const pBufferDst, void const * const pBufferSrc, std::size_t const & uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    pageLockHostMem(pBufferDst, uiSizeBytes);

                    ALPAKA_CUDA_CHECK(cudaMemcpy(pBufferDst, pBufferSrc, uiSizeBytes, cudaMemcpyDeviceToHost));

                    unPageLockHostMem(pBufferSrc);
                }
            };
            //#############################################################################
            //! The CUDA memory copier.
            //!
            //! Copies from Host memory into CUDA memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceCuda, MemorySpaceHost>
            {
                MemCopy(void * const pBufferDst, void const * const pBufferSrc, std::size_t const & uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    pageLockHostMem(pBufferSrc, uiSizeBytes);

                    ALPAKA_CUDA_CHECK(cudaMemcpy(pBufferDst, pBufferSrc, uiSizeBytes, cudaMemcpyHostToDevice));

                    unPageLockHostMem(pBufferSrc);
                }
            };
            //#############################################################################
            //! The CUDA memory copier.
            //!
            //! Copies from CUDA memory into CUDA memory.
            //#############################################################################
            template<>
            struct MemCopy<MemorySpaceCuda, MemorySpaceCuda>
            {
                MemCopy(void * const pBufferDst, void const * const pBufferSrc, std::size_t const & uiSizeBytes)
                {
                    assert(pBufferDst);
                    assert(pBufferSrc);

                    ALPAKA_CUDA_CHECK(cudaMemcpy(pBufferDst, pBufferSrc, uiSizeBytes, cudaMemcpyDeviceToDevice));
                }
            };

            //#############################################################################
            //! The CUDA memory setter.
            //#############################################################################
            template<>
            struct MemSet<MemorySpaceCuda>
            {
                MemSet(void * const pBuffer, int const iValue, std::size_t const & uiSizeBytes)
                {
                    assert(pBuffer);

                    ALPAKA_CUDA_CHECK(cudaMemset(pBuffer, iValue, uiSizeBytes));
                }
            };
        }
    }
}
