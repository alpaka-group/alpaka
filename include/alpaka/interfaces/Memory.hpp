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

#include <cassert>  // assert
#include <memory>   // std::shared_ptr

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The memory management functionality.
    //-----------------------------------------------------------------------------
    namespace memory
    {
        namespace detail
        {
            //#############################################################################
            //! The abstract memory allocator.
            //#############################################################################
            template<typename TMemorySpace>
            struct MemAlloc;

            //#############################################################################
            //! The abstract memory freer.
            //#############################################################################
            template<typename TMemorySpace>
            struct MemFree;

            //#############################################################################
            //! The abstract memory copier.
            //!
            //! Copies memory from one buffer into another buffer possibly in a different memory space.
            //#############################################################################
            template<typename TMemorySpaceDst, typename TMemorySpaceSrc>
            struct MemCopy;

            //#############################################################################
            //! The abstract memory setter.
            //!
            //! Fills the buffer with data.
            //#############################################################################
            template<typename TMemorySpace>
            struct MemSet;
        }

        //-----------------------------------------------------------------------------
        //! Allocates memory in the given memory space.
        //!
        //! \tparam TMemorySpace The memory space to allocate in.
        //! \tparam T The type of the returned buffer.
        //! \param uiSizeBytes Size in bytes to set.
        //! \return Pointer to newly allocated memory.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpace, typename T = void>
        std::shared_ptr<T> alloc(size_t const uiSizeBytes)
        {
            assert(uiSizeBytes>0);

            void * pBuffer(nullptr);
            detail::MemAlloc<TMemorySpace>(
                &pBuffer, 
                uiSizeBytes);
            assert(pBuffer);

            return 
                std::shared_ptr<T>(
                    reinterpret_cast<T *>(pBuffer), 
                    [=](T * b){detail::MemFree<TMemorySpace>{b};});
        }

        //-----------------------------------------------------------------------------
        //! Copies memory possibly between different memory spaces.
        //!
        //! \param pBufferDst Pointer to destination memory.
        //! \param pBufferSrc Pointer to source memory.
        //! \param uiSizeBytes Size in bytes to copy.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpaceDst, typename TMemorySpaceSrc, typename T>
        void copy(T * const pBufferDst, T * const pBufferSrc, std::size_t const & uiSizeBytes)
        {
            assert(pBufferDst);
            assert(pBufferSrc);

            detail::MemCopy<TMemorySpaceDst, TMemorySpaceSrc>(
                reinterpret_cast<void *>(pBufferDst), 
                reinterpret_cast<void *>(pBufferSrc), 
                uiSizeBytes);
        }

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value.
        //!
        //! \param pBuffer Pointer to memory.
        //! \param iValue Value to set for each byte of specified memory.
        //! \param uiSizeBytes Size in bytes to set.
        //-----------------------------------------------------------------------------
        template<typename TMemorySpace, typename T>
        void set(T * const pBuffer, int const iValue, std::size_t const & uiSizeBytes)
        {
            assert(pBuffer);

            detail::MemSet<TMemorySpace>(
                reinterpret_cast<void *>(pBuffer),
                iValue, 
                uiSizeBytes);
        }
    }
}
