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

#include <alpaka/host/MemorySpace.hpp>  // MemSpaceHost
#include <alpaka/cuda/MemorySpace.hpp>  // MemSpaceCuda

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
                typename T,
                typename TSfinae = void>
            struct GetMemSpace;

            //#############################################################################
            //! The memory element type trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetMemElem;

            //#############################################################################
            //! The memory buffer type trait.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim,
                typename TSfinae = void>
            struct GetMemBuf;

            //#############################################################################
            //! The native pointer get trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetNativePtr;

            //#############################################################################
            //! The pitch in bytes. This is the distance between two consecutive rows.
            //#############################################################################
            template<
                typename TMemBuf,
                typename TSfinae = void>
            struct GetPitchBytes;

            //#############################################################################
            //! The memory allocator trait.
            //#############################################################################
            template<
                typename TElem, 
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
    //! The memory trait accessors.
    //-----------------------------------------------------------------------------
    namespace memory
    {
        //#############################################################################
        //! The memory space trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using GetMemSpaceT = typename traits::memory::GetMemSpace<T>::type;

        //#############################################################################
        //! The memory element type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TMemBuf>
        using GetMemElemT = typename traits::memory::GetMemElem<TMemBuf>::type;

        //#############################################################################
        //! The memory buffer type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TElem,
            typename TDim,
            typename TMemSpace>
        using GetMemBufT = typename traits::memory::GetMemBuf<TElem, TDim, TMemSpace>::type;

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
            -> GetMemElemT<TMemBuf> const *
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
            -> GetMemElemT<TMemBuf> *
        {
            return traits::memory::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in bytes. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
            std::size_t getPitchBytes(
            TMemBuf const & memBuf)
        {
            return traits::memory::GetPitchBytes<TMemBuf>::getPitchBytes(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Allocates memory in the given memory space.
        //!
        //! \tparam T The type of the returned buffer.
        //! \tparam TMemSpace The memory space to allocate in.
        //! \param extent The extents of the buffer.
        //! \return Pointer to newly allocated buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TElem, 
            typename TMemSpace, 
            typename TExtent>
        ALPAKA_FCT_HOST auto alloc(
            TExtent const & extent = TExtent())
            -> decltype(traits::memory::MemAlloc<TElem, dim::GetDimT<TExtent>, TMemSpace>::memAlloc(std::declval<TElem>()))
        {
            return traits::memory::MemAlloc<TElem, dim::GetDimT<TExtent>, TMemSpace>::memAlloc(
                extent);
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
                "The destination buffer and the extent are required to have the same dimensionality!");
            static_assert(
                std::is_same<GetMemElemT<TMemBufDst>, GetMemElemT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            // \TODO: Copy of arrays of different dimensions. Maybe only 1D to ND?

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
            std::uint8_t const & byte, 
            TExtent const & extent = TExtent())
        {
            static_assert(
                std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                "The buffer and the extent are required to have the same dimensionality!");

            traits::memory::MemSet<dim::GetDimT<TMemBuf>, GetMemSpaceT<TMemBuf>>::memSet(
                memBuf,
                byte,
                extent);
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
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
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
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value >= 1u)
                    && (std::rank<TFixedSizeArray>::value <= 3u)
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value-1u>::value;
                }
            };

            //#############################################################################
            //! The fixed size array height get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetHeight<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value >= 2u)
                    && (std::rank<TFixedSizeArray>::value <= 3u)
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 2u>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getHeight(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 2u>::value;
                }
            };
            //#############################################################################
            //! The fixed size array depth get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetDepth<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value >= 3u)
                    && (std::rank<TFixedSizeArray>::value <= 3u)
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 3u>::value > 0u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getDepth(
                    TFixedSizeArray const &)
                {
                    return std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 3u>::value;
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
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
#ifdef __CUDA_ARCH__
                using type = alpaka::memory::MemSpaceCuda;
#else
                using type = alpaka::memory::MemSpaceHost;
#endif
            };

            //#############################################################################
            //! The fixed size array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetMemElem<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
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
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    TFixedSizeArray const & memBuf)
                {
                    return memBuf;
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    TFixedSizeArray & memBuf)
                {
                    return memBuf;
                }
            };

            //#############################################################################
            //! The fixed size array pitch get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetPitchBytes<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
            {
                using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                ALPAKA_FCT_HOST_ACC static std::size_t getPitchBytes(
                    TFixedSizeArray const &)
                {
                    return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
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
                typename TElem,
                std::size_t TSize>
            struct GetDim<
                std::array<TElem, TSize>>
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
                typename TElem,
                std::size_t TSize>
            struct GetWidth<
                std::array<TElem, TSize> >
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    std::array<TElem, TSize> const & extent)
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
                typename TElem,
                std::size_t TSize>
            struct GetMemSpace<
                std::array<TElem, TSize>>
            {
                using type = alpaka::memory::MemSpaceHost;
            };

            //#############################################################################
            //! The std::array memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TSize>
            struct GetMemElem<
                std::array<TElem, TSize>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The std::array native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TSize>
            struct GetNativePtr<
                std::array<TElem, TSize>>
            {
                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    std::array<TElem, TSize> const & memBuf)
                {
                    return memBuf.data();
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    std::array<TElem, TSize> & memBuf)
                {
                    return memBuf.data();
                }
            };

            //#############################################################################
            //! The std::array pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TSize>
            struct GetPitchBytes<
                std::array<TElem, TSize>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getPitchBytes(
                    std::array<TElem, TSize> const & memPitch)
                {
                    return sizeof(TElem) * memPitch.size();
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
                typename TElem,
                typename Allocator>
            struct GetDim<
                std::vector<TElem, Allocator>>
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
                typename TElem,
                typename Allocator>
            struct GetWidth<
                std::vector<TElem, Allocator>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    std::vector<TElem, Allocator> const & extent)
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
                typename TElem,
                typename Allocator>
            struct GetMemSpace<
                std::vector<TElem, Allocator>>
            {
                using type = alpaka::memory::MemSpaceHost;
            };

            //#############################################################################
            //! The std::vector memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetMemElem<
                std::vector<TElem, Allocator>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The std::vector native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetNativePtr<
                std::vector<TElem, Allocator>>
            {
                ALPAKA_FCT_HOST_ACC static TElem const * getNativePtr(
                    std::vector<TElem, Allocator> const & memBuf)
                {
                    return memBuf.data();
                }
                ALPAKA_FCT_HOST_ACC static TElem * getNativePtr(
                    std::vector<TElem, Allocator> & memBuf)
                {
                    return memBuf.data();
                }
            };

            //#############################################################################
            //! The std::vector pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename Allocator>
            struct GetPitchBytes<
                std::vector<TElem, Allocator>>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getPitchBytes(
                    std::vector<TElem, Allocator> const & memPitch)
                {
                    return sizeof(TElem) * memPitch.size();
                }
            };
        }
    }
}
