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

#include <cassert>  // assert

namespace alpaka
{
    namespace memory
    {
        namespace detail
        {
            //#############################################################################
            //! Allocates memory in the given memory space.
            //#############################################################################
            template<typename TMemorySpace>
            struct MemAlloc;

            //#############################################################################
            //! Frees memory in the given memory space.
            // TODO: Remove this function and replace by a RAII wrapper for the memory?
            //#############################################################################
            template<typename TMemorySpace>
            struct MemFree;

            //#############################################################################
            //! Copies memory from one buffer into another buffer possibly on different memory space.
            //#############################################################################
            template<typename TMemorySpaceDst, typename TDtaSpaceSrc>
            struct MemCopy;

            //#############################################################################
            //! Fills the buffer with the data in the given memory space.
            //#############################################################################
            template<typename TMemorySpace>
            struct MemSet;
        }

        //-----------------------------------------------------------------------------
        //! \tparam TMemorySpace The memory space to allocate in.
        //! \tparam T The type of the returned buffer.
        //! \param uiSizeBytes Size in bytes to set.
        //! \return Pointer to newly allocated memory.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpace, typename T = void>
        T * memAlloc(size_t const uiSizeBytes)
        {
            assert(uiSizeBytes>0);

            void * pBuffer(nullptr);
            detail::MemAlloc<TMemorySpace>(
                &pBuffer, 
                uiSizeBytes);
            assert(pBuffer);
            return reinterpret_cast<T *>(pBuffer);
        }

        //-----------------------------------------------------------------------------
        //! \param pBuffer Pointer to memory to free.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpace, typename T = void>
        void memFree(T * pBuffer)
        {
            assert(reinterpret_cast<void *>(pBuffer));

            detail::MemFree<TMemorySpace>{pBuffer};
        }

        //-----------------------------------------------------------------------------
        //! \param pBufferDst Pointer to destination memory.
        //! \param pBufferSrc Pointer to source memory.
        //! \param uiSizeBytes Size in bytes to copy.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpaceDst, typename TDtaSpaceSrc, typename T = void>
        void memCopy(T * const pBufferDst, T * const pBufferSrc, size_t const uiSizeBytes)
        {
            assert(pBufferDst);
            assert(pBufferSrc);

            detail::MemCopy<TMemorySpaceDst, TDtaSpaceSrc>(
                reinterpret_cast<void *>(pBufferDst), 
                reinterpret_cast<void *>(pBufferSrc), 
                uiSizeBytes);
        }

        //-----------------------------------------------------------------------------
        //! \param pBuffer Pointer to memory.
        //! \param iValue Value to set for each byte of specified memory.
        //! \param uiSizeBytes Size in bytes to set.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpace, typename T = void>
        void memSet(T * const pBuffer, int const iValue, size_t const uiSizeBytes)
        {
            assert(pBuffer);

            detail::MemSet<TMemorySpace>(
                reinterpret_cast<void *>(pBuffer),
                iValue, 
                uiSizeBytes);
        }
    }
}
