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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

#include <alpaka/traits/Dim.hpp>        // GetDimT
#include <alpaka/traits/Extent.hpp>     // traits::getXXX

#include <alpaka/host/MemorySpace.hpp>  // MemorySpaceHost
#include <alpaka/cuda/MemorySpace.hpp>  // MemorySpaceCuda

#include <type_traits>                  // std::enable_if, std::is_array, std::extent
#include <vector>                       // std::vector
#include <array>                        // std::array

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The memory management traits.
        //-----------------------------------------------------------------------------
        namespace memory
        {
            //#############################################################################
            //! The memory space trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetMemSpace;

            //#############################################################################
            //! The memory element type trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetMemElemType;

            //#############################################################################
            //! The native pointer get trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetNativePtr;

            //#############################################################################
            //! The memory allocator trait.
            //#############################################################################
            template<
                typename TElement, 
                typename TDim, 
                typename TMemSpace, 
                typename TSfinae = void>
            struct MemAlloc;

            //#############################################################################
            //! The memory copy trait.
            //!
            //! Copies memory from one buffer into another buffer possibly in a different memory space.
            //#############################################################################
            template<
                typename TDim, 
                typename TMemSpaceDst, 
                typename TMemSpaceSrc, 
                typename TSfinae = void>
            struct MemCopy;

            //#############################################################################
            //! The memory set trait.
            //!
            //! Fills the buffer with data.
            //#############################################################################
            template<
                typename TDim, 
                typename TMemSpace, 
                typename TSfinae = void>
            struct MemSet;
        }
    }

    //-----------------------------------------------------------------------------
    //! The memory management trait accessors.
    //-----------------------------------------------------------------------------
    namespace memory
    {
        //#############################################################################
        //! The memory space trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TMemBuf>
        using GetMemSpaceT = typename traits::memory::GetMemSpace<TMemBuf>::type;

        //#############################################################################
        //! The memory element type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TMemBuf>
        using GetMemElemTypeT = typename traits::memory::GetMemElemType<TMemBuf>::type;

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TMemBuf const & memBuf)
            -> GetMemElemTypeT<TMemBuf> const *
        {
            return traits::memory::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TMemBuf & memBuf)
            -> GetMemElemTypeT<TMemBuf> *
        {
            return traits::memory::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Allocates memory in the given memory space.
        //!
        //! \tparam T The type of the returned buffer.
        //! \tparam TMemSpace The memory space to allocate in.
        //! \tparam TMemLayout The type holding the memory layout of the buffer (pitches).
        //! \param extent The extents of the buffer.
        //! \param memLayout The memory layout of the buffer (pitches).
        //! \return Pointer to newly allocated buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TElement, 
            typename TMemSpace, 
            typename TExtent/*, 
            typename TMemLayout*/>
        ALPAKA_FCT_HOST auto alloc(
            TExtent const & extent = TExtent()/*, 
            TMemLayout const & memLayout*/)
            -> decltype(traits::memory::MemAlloc<TElement, dim::GetDimT<TExtent>, TMemSpace>::memAlloc(std::declval<TElement>()/*, memLayout*/))
        {
            /*static_assert(
                std::is_same<GetDimT<TExtent>, GetDimT<TMemLayout>>::value,
                "The extent and the memLayout are required to have the same dimensionality!");*/

            return traits::memory::MemAlloc<TElement, dim::GetDimT<TExtent>, TMemSpace>::memAlloc(
                extent/*, 
                memLayout*/);
        }

        //-----------------------------------------------------------------------------
        //! Copies memory possibly between different memory spaces.
        //!
        //! \param memBufDst The destination memory buffer.
        //! \param memBufSrc The source memory buffer.
        //! \param extent The extents of the buffer to copy.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBufDst, 
            typename TMemBufSrc, 
            typename TExtent>
        ALPAKA_FCT_HOST void copy(
            TMemBufDst & memBufDst, 
            TMemBufSrc const & memBufSrc, 
            TExtent const & extent = TExtent())
        {
            static_assert(
                std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtent>>::value,
                "The buffers and the extent are required to have the same dimensionality!");
            static_assert(
                std::is_same<GetMemElemTypeT<TMemBufDst>, GetMemElemTypeT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::memory::MemCopy<dim::GetDimT<TMemBufDst>, GetMemSpaceT<TMemBufDst>, GetMemSpaceT<TMemBufSrc>>::memCopy(
                memBufDst,
                memBufSrc,
                extent);
        }

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \param value Value to set for each element of the specified buffer.
        //! \param extent The extents of the buffer to fill.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf, 
            typename TExtent>
        ALPAKA_FCT_HOST void set(
            TMemBuf & memBuf, 
            int const & iValue, 
            TExtent const & extent = TExtent())
        {
            static_assert(
                std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                "The destination buffer and the extent are required to have the same dimensionality!");

            traits::memory::MemSet<dim::GetDimT<TMemBuf>, GetMemSpaceT<TMemBuf>>::memSet(
                memBuf,
                iValue,
                extent);
        }
    }
    

    //-----------------------------------------------------------------------------
    // Trait specializations for compile time arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory operations.
    // Up to 3 dimensions are supported.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The compile time array dimension getter trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetDim<
                T,
                typename std::enable_if<std::is_array<T>::value, void>::type>
            {
                using type = alpaka::dim::Dim<std::rank<T>::value>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The compile time array width get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetWidth<
                T,
                typename std::enable_if<std::is_array<T>::value && (std::rank<T>::value >= 1) && (std::rank<T>::value <= 3), void>::type>
            {
                static std::size_t getWidth(
                    T const &)
                {
                    return std::extent<T, std::rank<T>::value-1>::value;
                }
            };

            //#############################################################################
            //! The compile time array height get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetHeight<
                T,
                typename std::enable_if<std::is_array<T>::value && (std::rank<T>::value >= 2) && (std::rank<T>::value <= 3), void>::type>
            {
                static std::size_t getHeight(
                    T const &)
                {
                    return std::extent<T, std::rank<T>::value - 2>::value;
                }
            };
            //#############################################################################
            //! The compile time array depth get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetDepth<
                T,
                typename std::enable_if<std::is_array<T>::value && (std::rank<T>::value >= 3) && (std::rank<T>::value <= 3), void>::type>
            {
                static std::size_t getDepth(
                    T const &)
                {
                    return std::extent<T, std::rank<T>::value - 3>::value;
                }
            };
        }

        namespace memory
        {
            //#############################################################################
            //! The compile time array memory space trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetMemSpace<
                T,
                typename std::enable_if<std::is_array<T>::value, void>::type>
            {
#ifdef __CUDA_ARCH__
                using type = MemorySpaceCuda;
#else
                using type = MemorySpaceHost;
#endif
            };

            //#############################################################################
            //! The compile time array memory element type get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetMemElemType<
                T,
                typename std::enable_if<std::is_array<T>::value, void>::type>
            {
                using type = typename std::remove_all_extents<T>::type;
            };

            //#############################################################################
            //! The compile time array native pointer get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetNativePtr<
                T,
                typename std::enable_if<std::is_array<T>::value, void>::type>
            {
                static T const * getNativePtr(
                    T const & memBuf)
                {
                    return memBuf;
                }
                static T * getNativePtr(
                    T & memBuf)
                {
                    return memBuf;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::array.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The std::array dimension getter trait specialization.
            //#############################################################################
            template<
                typename T, 
                std::size_t TSize>
            struct GetDim<
                std::array<T, TSize>>
            {
                using type = alpaka::dim::Dim1;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The std::array width get trait specialization.
            //#############################################################################
            template<
                typename T, 
                std::size_t TSize>
            struct GetWidth<
                std::array<T, TSize> >
            {
                static std::size_t getWidth(
                    std::array<T, TSize> const & extent)
                {
                    return extent.size();
                }
            };
        }

        namespace memory
        {
            //#############################################################################
            //! The std::array memory space trait specialization.
            //#############################################################################
            template<
                typename T, 
                std::size_t TSize>
            struct GetMemSpace<
                std::array<T, TSize>>
            {
                using type = MemorySpaceHost;
            };

            //#############################################################################
            //! The std::array memory element type get trait specialization.
            //#############################################################################
            template<
                typename T, 
                std::size_t TSize>
            struct GetMemElemType<
                std::array<T, TSize>>
            {
                using type = T;
            };

            //#############################################################################
            //! The std::array native pointer get trait specialization.
            //#############################################################################
            template<
                typename T, 
                std::size_t TSize>
            struct GetNativePtr<
                std::array<T, TSize>>
            {
                static T const * getNativePtr(
                    std::array<T, TSize> const & memBuf)
                {
                    return memBuf.data();
                }
                static T * getNativePtr(
                    std::array<T, TSize> & memBuf)
                {
                    return memBuf.data();
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::vector.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The dimension getter trait.
            //#############################################################################
            template<
                typename T, 
                typename Allocator>
            struct GetDim<
                std::vector<T, Allocator>>
            {
                using type = alpaka::dim::Dim1;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The std::vector width get trait specialization.
            //#############################################################################
            template<
                typename T, 
                typename Allocator>
            struct GetWidth<
                std::vector<T, Allocator>>
            {
                static std::size_t getWidth(
                    std::vector<T, Allocator> const & extent)
                {
                    return extent.size();
                }
            };
        }

        namespace memory
        {
            //#############################################################################
            //! The std::vector memory space trait specialization.
            //#############################################################################
            template<
                typename T, 
                typename Allocator>
            struct GetMemSpace<
                std::vector<T, Allocator>>
            {
                using type = MemorySpaceHost;
            };

            //#############################################################################
            //! The std::vector memory element type get trait specialization.
            //#############################################################################
            template<
                typename T, 
                typename Allocator>
            struct GetMemElemType<
                std::vector<T, Allocator>>
            {
                using type = T;
            };

            //#############################################################################
            //! The std::vector native pointer get trait specialization.
            //#############################################################################
            template<
                typename T, 
                typename Allocator>
            struct GetNativePtr<
                std::vector<T, Allocator>>
            {
                static T const * getNativePtr(
                    std::vector<T, Allocator> const & memBuf)
                {
                    return memBuf.data();
                }
                static T * getNativePtr(
                    std::vector<T, Allocator> & memBuf)
                {
                    return memBuf.data();
                }
            };
        }
    }

    //#############################################################################
    //! The runtime memory layout interface.
    //!
    //! This defines the pitches of the memory buffer.
    //#############################################################################
    /*template<
        typename TDim>
    struct RuntimeMemLayout;

    //#############################################################################
    //! The 1D runtime memory layout.
    //#############################################################################
    template<>
    struct RuntimeMemLayout<
        Dim1>
    {};
    //#############################################################################
    //! The 2D runtime memory layout.
    //#############################################################################
    template<>
    struct RuntimeMemLayout<
        Dim2>
    {
        std::size_t uiRowPitchBytes;    //!< The width in bytes of the 2D array pointed to, including any padding added to the end of each row.
    };
    //#############################################################################
    //! The 3D runtime memory layout.
    //#############################################################################
    template<>
    struct RuntimeMemLayout<
        Dim3>
    {
        std::size_t uiRowPitchBytes;    //!< The width in bytes of the 3D array pointed to, including any padding added to the end of each row.
        std::size_t uiRowWidthBytes;    //!< The width of each row in bytes.
        std::size_t uiSliceHeightRows;  //!< The height of each 2D slice in rows.
    };*/
}
