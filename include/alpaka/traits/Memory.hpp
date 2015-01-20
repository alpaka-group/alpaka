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
        //! The memory traits.
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

            //-----------------------------------------------------------------------------
            //! The memory layout traits.
            //-----------------------------------------------------------------------------
            namespace layout
            {
                //#############################################################################
                //! The pitch in bytes. This is the distance between two consecutive rows.
                //#############################################################################
                template<
                    typename TMemBuf,
                    typename TSfinae = void>
                struct GetPitchBytes;
            }
        }
    }

    //-----------------------------------------------------------------------------
    //! The memory trait accessors.
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

        //-----------------------------------------------------------------------------
        //! The memory layout traits specializations.
        //-----------------------------------------------------------------------------
        namespace layout
        {
            //-----------------------------------------------------------------------------
            //! \return The pitch in bytes. This is the distance between two consecutive rows.
            //-----------------------------------------------------------------------------
            template<
                typename TMemBuf>
            std::size_t getPitchBytes()
            {
                return traits::memory::layout::GetPitchBytes<TMemBuf>::getPitchBytes();
            }
        }
    }
    

    //-----------------------------------------------------------------------------
    // Trait specializations for fixed size arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory operations.
    // Up to 3 dimensions are supported.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The fixed size array dimension getter trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetDim<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value, void>::type>
            {
                using type = alpaka::dim::Dim<std::rank<TFixedSizeArray>::value>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The fixed size array width get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetWidth<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value && (std::rank<TFixedSizeArray>::value >= 1) && (std::rank<TFixedSizeArray>::value <= 3), void>::type>
            {
                static std::size_t getWidth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value-1>::value;
                }
            };

            //#############################################################################
            //! The fixed size array height get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetHeight<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value && (std::rank<TFixedSizeArray>::value >= 2) && (std::rank<TFixedSizeArray>::value <= 3), void>::type>
            {
                static std::size_t getHeight(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 2>::value;
                }
            };
            //#############################################################################
            //! The fixed size array depth get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetDepth<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value && (std::rank<TFixedSizeArray>::value >= 3) && (std::rank<TFixedSizeArray>::value <= 3), void>::type>
            {
                static std::size_t getDepth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 3>::value;
                }
            };
        }

        namespace memory
        {
            //#############################################################################
            //! The fixed size array memory space trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetMemSpace<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value, void>::type>
            {
#ifdef __CUDA_ARCH__
                using type = MemorySpaceCuda;
#else
                using type = MemorySpaceHost;
#endif
            };

            //#############################################################################
            //! The fixed size array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetMemElemType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value, void>::type>
            {
                using type = typename std::remove_all_extents<TFixedSizeArray>::type;
            };

            //#############################################################################
            //! The fixed size array native pointer get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetNativePtr<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value, void>::type>
            {
                using TElement = typename std::remove_all_extents<TFixedSizeArray>::type;

                static TElement const * getNativePtr(
                    TFixedSizeArray const & memBuf)
                {
                    return memBuf;
                }
                static TElement * getNativePtr(
                    TFixedSizeArray & memBuf)
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
                typename TElement,
                std::size_t TSize>
            struct GetDim<
                std::array<TElement, TSize>>
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
                typename TElement,
                std::size_t TSize>
            struct GetWidth<
                std::array<TElement, TSize> >
            {
                static std::size_t getWidth(
                    std::array<TElement, TSize> const & extent)
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
                typename TElement,
                std::size_t TSize>
            struct GetMemSpace<
                std::array<TElement, TSize>>
            {
                using type = MemorySpaceHost;
            };

            //#############################################################################
            //! The std::array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElement,
                std::size_t TSize>
            struct GetMemElemType<
                std::array<TElement, TSize>>
            {
                using type = TElement;
            };

            //#############################################################################
            //! The std::array native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElement,
                std::size_t TSize>
            struct GetNativePtr<
                std::array<TElement, TSize>>
            {
                static TElement const * getNativePtr(
                    std::array<TElement, TSize> const & memBuf)
                {
                    return memBuf.data();
                }
                static TElement * getNativePtr(
                    std::array<TElement, TSize> & memBuf)
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
                typename TElement,
                typename Allocator>
            struct GetDim<
                std::vector<TElement, Allocator>>
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
                typename TElement,
                typename Allocator>
            struct GetWidth<
                std::vector<TElement, Allocator>>
            {
                static std::size_t getWidth(
                    std::vector<TElement, Allocator> const & extent)
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
                typename TElement,
                typename Allocator>
            struct GetMemSpace<
                std::vector<TElement, Allocator>>
            {
                using type = MemorySpaceHost;
            };

            //#############################################################################
            //! The std::vector memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElement,
                typename Allocator>
            struct GetMemElemType<
                std::vector<TElement, Allocator>>
            {
                using type = TElement;
            };

            //#############################################################################
            //! The std::vector native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElement,
                typename Allocator>
            struct GetNativePtr<
                std::vector<TElement, Allocator>>
            {
                static TElement const * getNativePtr(
                    std::vector<TElement, Allocator> const & memBuf)
                {
                    return memBuf.data();
                }
                static TElement * getNativePtr(
                    std::vector<TElement, Allocator> & memBuf)
                {
                    return memBuf.data();
                }
            };
        }
    }
}
