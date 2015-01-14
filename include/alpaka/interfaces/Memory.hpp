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

#include <cassert>                      // assert
#include <memory>                       // std::shared_ptr

#include <alpaka/core/Dimension.hpp>    // Dim<N>

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
            template<typename T, typename TDim, typename TMemorySpace>
            struct MemAlloc;

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

        //#############################################################################
        //! The runtime memory layout interface.
        //!
        //! This defines the pitches of the memory buffer.
        //#############################################################################
        /*template<typename TDim>
        struct RuntimeMemLayout;

        //#############################################################################
        //! The 1D runtime memory layout.
        //#############################################################################
        template<>
        struct RuntimeMemLayout<Dim1>
        {};
        //#############################################################################
        //! The 2D runtime memory layout.
        //#############################################################################
        template<>
        struct RuntimeMemLayout<Dim2>
        {
            std::size_t uiRowPitchBytes;    //!< The width in bytes of the 2D array pointed to, including any padding added to the end of each row.
        };
        //#############################################################################
        //! The 3D runtime memory layout.
        //#############################################################################
        template<>
        struct RuntimeMemLayout<Dim3>
        {
            std::size_t uiRowPitchBytes;    //!< The width in bytes of the 3D array pointed to, including any padding added to the end of each row.
            std::size_t uiRowWidthBytes;    //!< The width of each row in bytes.
            std::size_t uiSliceHeightRows;  //!< The height of each 2D slice in rows.
        };


        //#############################################################################
        //! The memory buffer interface.
        //!
        //! Templating on the memory space prevents mixing pointers of different memory spaces.
        //#############################################################################
        template</*typename TMemLayout, *//*typename TMemExtent, typename TMemAllocator, typename TDim, typename TMemorySpace>
        class Buffer :
            public TMemAllocator,
            //public TMemLayout,
            public TMemExtent
        {
        public:
            using MemorySpace = TMemorySpace;
            using Dim = TDim;

        private:
            void * pMem;  //!< The pointer to the memory.
        };

        namespace detail
        {
            //#############################################################################
            //! The abstract native pointer getter.
            //#############################################################################
            template<typename TMemoryObject>
            struct GetNativePtr();

            //#############################################################################
            //! The simple pointer native pointer getter.
            //#############################################################################
            template<typename TMemoryObject>
            struct GetNativePtr<T*>
            {
                T * operator()(TMemoryObject const & memObject)
                {
                    return memObject.pMem;
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<typename TMemoryObject>
        auto getNativePtr(TMemoryObject const & memObject)
            -> typename std::result_of<detail::GetNativePtr()(TMemoryObject)>::type *
        {
            return detail::GetNativePtr()(memObject);
        }*/

        //-----------------------------------------------------------------------------
        //! Allocates memory in the given memory space.
        //!
        //! \tparam T The type of the returned buffer.
        //! \tparam TMemorySpace The memory space to allocate in.
        //! \tparam TMemExtent The type holding the extents of the buffer.
        //! \tparam TMemLayout The type holding the memory layout of the buffer (pitches).
        //! \param memExtent The extents of the buffer.
        //! \param memLayout The memory layout of the buffer (pitches).
        //! \return Pointer to newly allocated buffer.
        //-----------------------------------------------------------------------------
        template<typename T, typename TMemorySpace, typename TMemExtent/*, typename TMemLayout*/>
        auto alloc(TMemExtent const & memExtent/*, TMemLayout const & memLayout*/)
            -> typename std::result_of<detail::MemAlloc<T, dim::GetDimT<TMemExtent>, TMemorySpace>()(TMemExtent/*, TMemLayout*/)>::type
        {
            /*static_assert(
                std::is_same<GetDimT<TMemExtent>, GetDimT<TMemLayout>>, 
                "The memExtent and the memLayout are required to have the same dimensionality!");*/

            return detail::MemAlloc<T, dim::GetDimT<TMemExtent>, TMemorySpace>()(memExtent/*, memLayout*/);
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
