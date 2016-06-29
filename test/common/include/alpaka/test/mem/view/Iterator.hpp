/**
* \file
* Copyright 2014-2016 Erik Zenker
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

#include <alpaka/alpaka.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    //-----------------------------------------------------------------------------
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test mem specifics.
        //-----------------------------------------------------------------------------
        namespace mem
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            namespace view
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                namespace traits
                {
                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    struct IteratorType
                    {
                        using Dim  = alpaka::dim::Dim<TView>;
                        using Size = alpaka::size::Size<TView>;
                        using Elem = alpaka::elem::Elem<TView>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST IteratorType(
                            TView & view,
                            Size const idx) :
                                m_nativePtr(alpaka::mem::view::getPtrNative(view)),
                                m_currentIdx(idx),
                                m_extents(alpaka::extent::getExtentVec(view)),
                                m_pitchBytes(alpaka::mem::view::getPitchBytesVec(view))
                        {}

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST IteratorType(
                            TView & view) :
                                IteratorType(view, 0)
                        {}

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator++()
                        -> IteratorType&
                        {
                            ++m_currentIdx;
                            return *this;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator--()
                        -> IteratorType&
                        {
                            --m_currentIdx;
                            return *this;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator++(
                            int)
                        -> IteratorType
                        {
                            IteratorType iterCopy = *this;
                            m_currentIdx++;
                            return iterCopy;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator--(
                            int)
                        -> IteratorType
                        {
                            IteratorType iterCopy = *this;
                            m_currentIdx--;
                            return iterCopy;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        template<typename TIter>
                        ALPAKA_FN_HOST_ACC auto operator==(
                            TIter &other) const
                        -> bool
                        {
                            return m_currentIdx == other.m_currentIdx;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        template<typename TIter>
                        ALPAKA_FN_HOST_ACC auto operator!=(
                            TIter &other) const
                        -> bool
                        {
                            return m_currentIdx != other.m_currentIdx;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator*() const
                        -> Elem &
                        {
                            using Dim1 = dim::DimInt<1>;

                            Vec<Dim1, Size> const currentIdxDim1{m_currentIdx};
                            Vec<Dim, Size> const currentIdxDimx(core::mapIdx<Dim::value>(currentIdxDim1, m_extents));

                            Elem * ptr = m_nativePtr;

                            for(Size dim_i = 0; dim_i + 1 < static_cast<Size>(Dim::value); dim_i++)
                            {
                                ptr += (currentIdxDimx[dim_i] * m_pitchBytes[dim_i+1]) / sizeof(Elem);
                            }

                            ptr += currentIdxDimx[Dim::value - 1];

                            return *ptr;
                        }

                        Elem * m_nativePtr;
                        Size  m_currentIdx;
                        alpaka::Vec<Dim, Size> const m_extents;
                        alpaka::Vec<Dim, Size> const m_pitchBytes;
                    };

                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    struct Begin
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto begin(
                            TView & view)
                        -> IteratorType<TView>
                        {
                            return IteratorType<TView>(view);
                        }
                    };

                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    struct End
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto end(
                            TView & view)
                        -> IteratorType<TView>
                        {
                            auto extents = alpaka::extent::getExtentVec(view);
                            return IteratorType<TView>(view, extents.prod());
                        }
                    };
                }

                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TView>
                using Iterator = traits::IteratorType<TView>;


                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                ALPAKA_FN_HOST static auto begin(
                    TView & view)
                -> Iterator<TView>
                {
                    return traits::Begin<TView>::begin(view);
                }

                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                ALPAKA_FN_HOST static auto end(
                    TView & view)
                -> Iterator<TView>
                {
                    return traits::End<TView>::end(view);
                }
            }
        }
    }
}
