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
                    // \tparam T Type to conditionally make const.
                    // \tparam TSource Type to mimic the constness of.
                    //#############################################################################
                    template<
                        typename T,
                        typename TSource>
                    using MimicConst = typename std::conditional<
                        std::is_const<TSource>::value,
                        typename std::add_const<T>::type,
                        typename std::remove_const<T>::type>;

                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    class IteratorView
                    {
                        using TViewDecayed = typename std::decay<TView>::type;
                        using Dim = alpaka::dim::Dim<TViewDecayed>;
                        using Size = alpaka::size::Size<TViewDecayed>;
                        using Elem = typename MimicConst<alpaka::elem::Elem<TViewDecayed>, TView>::type;

                    public:
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST IteratorView(
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
                        ALPAKA_FN_HOST IteratorView(
                            TView & view) :
                                IteratorView(view, 0)
                        {}

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator++()
                        -> IteratorView&
                        {
                            ++m_currentIdx;
                            return *this;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator--()
                        -> IteratorView&
                        {
                            --m_currentIdx;
                            return *this;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator++(
                            int)
                        -> IteratorView
                        {
                            IteratorView iterCopy = *this;
                            m_currentIdx++;
                            return iterCopy;
                        }

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator--(
                            int)
                        -> IteratorView
                        {
                            IteratorView iterCopy = *this;
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

                            vec::Vec<Dim1, Size> const currentIdxDim1{m_currentIdx};
                            vec::Vec<Dim, Size> const currentIdxDimx(idx::mapIdx<Dim::value>(currentIdxDim1, m_extents));

                            Elem * ptr = m_nativePtr;

                            for(Size dim_i(0); dim_i + 1 < static_cast<Size>(Dim::value); ++dim_i)
                            {
                                ptr += static_cast<Size>(currentIdxDimx[dim_i] * m_pitchBytes[dim_i+1]) / static_cast<Size>(sizeof(Elem));
                            }

                            ptr += currentIdxDimx[Dim::value - 1];

                            return *ptr;
                        }

                    private:
                        Elem * const m_nativePtr;
                        Size m_currentIdx;
                        vec::Vec<Dim, Size> const m_extents;
                        vec::Vec<Dim, Size> const m_pitchBytes;
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
                        -> IteratorView<TView>
                        {
                            return IteratorView<TView>(view);
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
                        -> IteratorView<TView>
                        {
                            auto extents = alpaka::extent::getExtentVec(view);
                            return IteratorView<TView>(view, extents.prod());
                        }
                    };
                }

                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TView>
                using Iterator = traits::IteratorView<TView>;

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
