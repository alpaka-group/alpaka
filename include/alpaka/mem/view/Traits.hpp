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

#include <alpaka/dev/Traits.hpp>        // dev::DevType, ...
#include <alpaka/dim/Traits.hpp>        // dim::DimType
#include <alpaka/extent/Traits.hpp>     // extent::GetExtent
#include <alpaka/offset/Traits.hpp>     // offset::GetOffset

#include <alpaka/core/Fold.hpp>         // foldr
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

#include <iosfwd>                       // std::ostream

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The buffer specifics.
        //-----------------------------------------------------------------------------
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The memory element type trait.
                //#############################################################################
                template<
                    typename TView,
                    typename TSfinae = void>
                struct ElemType;

                //#############################################################################
                //! The native pointer get trait.
                //#############################################################################
                template<
                    typename TBuf,
                    typename TSfinae = void>
                struct GetPtrNative;

                //#############################################################################
                //! The pointer on device get trait.
                //#############################################################################
                template<
                    typename TBuf,
                    typename TDev,
                    typename TSfinae = void>
                struct GetPtrDev;

                //#############################################################################
                //! The pitch in bytes.
                //! This is the distance in bytes in the linear memory between two consecutive elements in the next higher dimension (TIdx+1).
                //!
                //! The default implementation uses the extent to calculate the pitch.
                //#############################################################################
                template<
                    typename TIdx,
                    typename TView,
                    typename TSfinae = void>
                struct GetPitchBytes
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPitchBytes(
                        TView const & view)
                    -> Uint
                    {
                        using IdxSequence = alpaka::detail::make_integer_sequence_start<Uint, TIdx::value, dim::DimT<TView>::value - TIdx::value>;
                        return
                            extentsProd(view, IdxSequence())
                            * sizeof(typename ElemType<TView>::type);
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        Uint... TIndices>
                    ALPAKA_FCT_HOST static auto extentsProd(
                        TView const & view,
                        alpaka::detail::integer_sequence<Uint, TIndices...> const &)
                    -> Uint
                    {
                        // For the case that the sequence is empty (index out of range), 1 is returned.
                        return alpaka::foldr(
                            std::multiplies<Uint>(),
                            1u,
                            extent::getExtent<TIndices, Uint>(view)...);
                    }
                };

                //#############################################################################
                //! The memory set trait.
                //!
                //! Fills the buffer with data.
                //#############################################################################
                template<
                    typename TDim,
                    typename TDev,
                    typename TSfinae = void>
                struct Set;

                //#############################################################################
                //! The memory copy trait.
                //!
                //! Copies memory from one buffer into another buffer possibly on a different device.
                //#############################################################################
                template<
                    typename TDim,
                    typename TDevDst,
                    typename TDevSrc,
                    typename TSfinae = void>
                struct Copy;

                //#############################################################################
                //! The memory buffer view type trait.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev,
                    typename TSfinae = void>
                struct ViewType;

                //#############################################################################
                //! The memory buffer view creation type trait.
                //#############################################################################
                template<
                    typename TView,
                    typename TSfinae = void>
                struct CreateView;

                //#############################################################################
                //! The buffer trait.
                //#############################################################################
                template<
                    typename TView,
                    typename TSfinae = void>
                struct GetBuf;
            }

            //#############################################################################
            //! The memory element type trait alias template to remove the ::type.
            //#############################################################################
            template<
                typename TView>
            using ElemT = typename std::remove_volatile<typename traits::ElemType<TView>::type>::type;

            //-----------------------------------------------------------------------------
            //! Gets the native pointer of the memory buffer.
            //!
            //! \param buf The memory buffer.
            //! \return The native pointer.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf>
            ALPAKA_FCT_HOST auto getPtrNative(
                TBuf const & buf)
            -> ElemT<TBuf> const *
            {
                return traits::GetPtrNative<
                    TBuf>
                ::getPtrNative(
                    buf);
            }
            //-----------------------------------------------------------------------------
            //! Gets the native pointer of the memory buffer.
            //!
            //! \param buf The memory buffer.
            //! \return The native pointer.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf>
            ALPAKA_FCT_HOST auto getPtrNative(
                TBuf & buf)
            -> ElemT<TBuf> *
            {
                return traits::GetPtrNative<
                    TBuf>
                ::getPtrNative(
                    buf);
            }

            //-----------------------------------------------------------------------------
            //! Gets the pointer to the buffer on the given device.
            //!
            //! \param buf The memory buffer.
            //! \param dev The device.
            //! \return The pointer on the device.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf,
                typename TDev>
            ALPAKA_FCT_HOST auto getPtrDev(
                TBuf const & buf,
                TDev const & dev)
            -> ElemT<TBuf> const *
            {
                return traits::GetPtrDev<
                    TBuf,
                    TDev>
                ::getPtrDev(
                    buf,
                    dev);
            }
            //-----------------------------------------------------------------------------
            //! Gets the pointer to the buffer on the given device.
            //!
            //! \param buf The memory buffer.
            //! \param dev The device.
            //! \return The pointer on the device.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf,
                typename TDev>
            ALPAKA_FCT_HOST auto getPtrDev(
                TBuf & buf,
                TDev const & dev)
            -> ElemT<TBuf> *
            {
                return traits::GetPtrDev<
                    TBuf,
                    TDev>
                ::getPtrDev(
                    buf,
                    dev);
            }

            //-----------------------------------------------------------------------------
            //! \return The pitch in bytes. This is the distance between two consecutive rows.
            //-----------------------------------------------------------------------------
            template<
                Uint TuiIdx,
                typename TVal,
                typename TView>
            ALPAKA_FCT_HOST auto getPitchBytes(
                TView const & buf)
            -> TVal
            {
                return
                    static_cast<TVal>(
                        traits::GetPitchBytes<
                            std::integral_constant<Uint, TuiIdx>,
                            TView>
                        ::getPitchBytes(
                            buf));
            }

            //-----------------------------------------------------------------------------
            //! Sets the memory to the given value.
            //!
            //! \param buf The memory buffer to fill.
            //! \param byte Value to set for each element of the specified buffer.
            //! \param extents The extents of the buffer to fill.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TExtents>
            ALPAKA_FCT_HOST auto set(
                TView & buf,
                std::uint8_t const & byte,
                TExtents const & extents)
            -> void
            {
                static_assert(
                    dim::DimT<TView>::value == dim::DimT<TExtents>::value,
                    "The buffer and the extents are required to have the same dimensionality!");

                traits::Set<
                    dim::DimT<TView>,
                    dev::DevT<TView>>
                ::set(
                    buf,
                    byte,
                    extents);
            }

            //-----------------------------------------------------------------------------
            //! Sets the memory to the given value asynchronously.
            //!
            //! \param buf The memory buffer to fill.
            //! \param byte Value to set for each element of the specified buffer.
            //! \param extents The extents of the buffer to fill.
            //! \param stream The stream to enqueue the buffer fill task into.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TExtents,
                typename TStream>
            ALPAKA_FCT_HOST auto set(
                TView & buf,
                std::uint8_t const & byte,
                TExtents const & extents,
                TStream const & stream)
            -> void
            {
                static_assert(
                    dim::DimT<TView>::value == dim::DimT<TExtents>::value,
                    "The buffer and the extents are required to have the same dimensionality!");

                traits::Set<
                    dim::DimT<TView>,
                    dev::DevT<TView>,
                    TStream>
                ::set(
                    buf,
                    byte,
                    extents,
                    stream);
            }

            //-----------------------------------------------------------------------------
            //! Copies memory possibly between different memory spaces.
            //!
            //! \param bufDst The destination memory buffer.
            //! \param bufSrc The source memory buffer.
            //! \param extents The extents of the buffer to copy.
            //-----------------------------------------------------------------------------
            template<
                typename TBufDst,
                typename TBufSrc,
                typename TExtents>
            ALPAKA_FCT_HOST auto copy(
                TBufDst & bufDst,
                TBufSrc const & bufSrc,
                TExtents const & extents)
            -> void
            {
                static_assert(
                    dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                    "The source and the destination buffers are required to have the same dimensionality!");
                static_assert(
                    dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                    "The destination buffer and the extents are required to have the same dimensionality!");
                static_assert(
                    std::is_same<ElemT<TBufDst>, typename std::remove_const<ElemT<TBufSrc>>::type>::value,
                    "The source and the destination buffers are required to have the same element type!");

                traits::Copy<
                    dim::DimT<TBufDst>,
                    dev::DevT<TBufDst>,
                    dev::DevT<TBufSrc>>
                ::copy(
                    bufDst,
                    bufSrc,
                    extents);
            }

            //-----------------------------------------------------------------------------
            //! Copies memory possibly between different memory spaces asynchronously.
            //!
            //! \param bufDst The destination memory buffer.
            //! \param bufSrc The source memory buffer.
            //! \param extents The extents of the buffer to copy.
            //! \param stream The stream to enqueue the buffer copy task into.
            //-----------------------------------------------------------------------------
            template<
                typename TBufDst,
                typename TBufSrc,
                typename TExtents,
                typename TStream>
            ALPAKA_FCT_HOST auto copy(
                TBufDst & bufDst,
                TBufSrc const & bufSrc,
                TExtents const & extents,
                TStream const & stream)
            -> void
            {
                static_assert(
                    dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                    "The source and the destination buffers are required to have the same dimensionality!");
                static_assert(
                    dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                    "The destination buffer and the extents are required to have the same dimensionality!");
                static_assert(
                    std::is_same<ElemT<TBufDst>, typename std::remove_const<ElemT<TBufSrc>>::type>::value,
                    "The source and the destination buffers are required to have the same element type!");

                traits::Copy<
                    dim::DimT<TBufDst>,
                    dev::DevT<TBufDst>,
                    dev::DevT<TBufSrc>>
                ::copy(
                    bufDst,
                    bufSrc,
                    extents,
                    stream);
            }

            //#############################################################################
            //! The memory buffer view type trait alias template to remove the ::type.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            using ViewT = typename traits::ViewType<TElem, TDim, TDev>::type;

            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf>
            ALPAKA_FCT_HOST auto createView(
                TBuf const & buf)
            -> decltype(traits::CreateView<TView>::createView(buf))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf);
            }
            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf>
            ALPAKA_FCT_HOST auto createView(
                TBuf & buf)
            -> decltype(traits::CreateView<TView>::createView(buf))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf);
            }
            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //! \param extentsElements The extents in elements.
            //! \param relativeOffsetsElements The offsets in elements.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf,
                typename TExtents,
                typename TOffsets>
            ALPAKA_FCT_HOST auto createView(
                TBuf const & buf,
                TExtents const & extentsElements,
                TOffsets const & relativeOffsetsElements = TOffsets())
            -> decltype(traits::CreateView<TView>::createView(buf, extentsElements, relativeOffsetsElements))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf,
                    extentsElements,
                    relativeOffsetsElements);
            }
            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //! \param extentsElements The extents in elements.
            //! \param relativeOffsetsElements The offsets in elements.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf,
                typename TExtents,
                typename TOffsets>
            ALPAKA_FCT_HOST auto createView(
                TBuf & buf,
                TExtents const & extentsElements,
                TOffsets const & relativeOffsetsElements = TOffsets())
            -> decltype(traits::CreateView<TView>::createView(buf, extentsElements, relativeOffsetsElements))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf,
                    extentsElements,
                    relativeOffsetsElements);
            }

            //-----------------------------------------------------------------------------
            //! Gets the memory buffer.
            //!
            //! \param view The object the buffer is received from.
            //! \return The memory buffer.
            //-----------------------------------------------------------------------------
            template<
                typename TView>
            ALPAKA_FCT_HOST auto getBuf(
                TView const & view)
            -> decltype(traits::GetBuf<TView>::getBuf(view))
            {
                return traits::GetBuf<
                    TView>
                ::getBuf(
                    view);
            }

            //-----------------------------------------------------------------------------
            //! Gets the memory buffer.
            //!
            //! \param view The object the buffer is received from.
            //! \return The memory buffer.
            //-----------------------------------------------------------------------------
            template<
                typename TView>
            ALPAKA_FCT_HOST auto getBuf(
                TView & view)
            -> decltype(traits::GetBuf<TView>::getBuf(view))
            {
                return traits::GetBuf<
                    TView>
                ::getBuf(
                    view);
            }
        }
    }
}
