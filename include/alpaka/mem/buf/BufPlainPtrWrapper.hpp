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

#include <alpaka/mem/buf/Traits.hpp>    // dev::traits::DevType, DimType, GetExtent,Copy, GetOffset, ...

#include <alpaka/vec/Vec.hpp>           // Vec<N>

namespace alpaka
{
    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The memory buffer wrapper used to wrap plain pointers.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            class BufPlainPtrWrapper final
            {
            public:
                using Dev = TDev;
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtents>
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(
                    TElem * pMem,
                    TDev const & dev,
                    TExtents const & extents = TExtents()) :
                        m_pMem(pMem),
                        m_dev(dev),
                        m_extentsElements(extent::getExtentsVecEnd<TDim>(extents)),
                        m_pitchBytes(extent::getWidth(extents) * sizeof(TElem))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtents>
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(
                    TElem * pMem,
                    TDev const dev,
                    TExtents const & extents,
                    TSize const & pitchBytes) :
                        m_pMem(pMem),
                        m_dev(dev),
                        m_extentsElements(extent::getExtentsVecEnd<TDim>(extents)),
                        m_pitchBytes(pitchBytes)
                {}

                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC auto operator=(BufPlainPtrWrapper const &) -> BufPlainPtrWrapper & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC auto operator=(BufPlainPtrWrapper &&) -> BufPlainPtrWrapper & = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC ~BufPlainPtrWrapper() = default;

            public:
                TElem * m_pMem;
                TDev m_dev;
                Vec<TDim, TSize> m_extentsElements;
                TSize m_pitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufPlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper device type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct DevType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper device get trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetDev<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getDev(
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> TDev
                {
                    return buf.m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct DimType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper memory element type get trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct ElemType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & extents)
                -> TSize
                {
                    return extents.m_extentsElements[TIdx::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufPlainPtrWrapper buf trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetBuf<
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getBuf(
                        mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getBuf(
                        mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> & buf)
                    -> mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> &
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The BufPlainPtrWrapper native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrNative<
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> TElem const *
                    {
                        return buf.m_pMem;
                    }
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> & buf)
                    -> TElem *
                    {
                        return buf.m_pMem;
                    }
                };

                //#############################################################################
                //! The BufPlainPtrWrapper memory pitch get trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPitchBytes(
                        mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> TSize
                    {
                        return buf.m_pitchBytes;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const &)
                -> TSize
                {
                    return 0u;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper size type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct SizeType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}