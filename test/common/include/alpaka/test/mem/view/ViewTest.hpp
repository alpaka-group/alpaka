/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

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
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize,
                    typename TDev,
                    typename TView>
                static auto viewTestImmutable(
                    TView const & view,
                    TDev const & dev,
                    alpaka::vec::Vec<TDim, TSize> const & extent,
                    alpaka::vec::Vec<TDim, TSize> const & offset)
                -> void
                {
                    //-----------------------------------------------------------------------------
                    // alpaka::dev::traits::DevType
                    {
                        static_assert(
                            std::is_same<alpaka::dev::Dev<TView>, TDev>::value,
                            "The device type of the view has to be equal to the specified one.");
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::dev::traits::GetDev
                    {
                        BOOST_REQUIRE(
                            dev == alpaka::dev::getDev(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::dim::traits::DimType
                    {
                        static_assert(
                            alpaka::dim::Dim<TView>::value == TDim::value,
                            "The dimensionality of the view has to be equal to the specified one.");
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::elem::traits::ElemType
                    {
                        static_assert(
                            std::is_same<alpaka::elem::Elem<TView>, TElem>::value,
                            "The element type of the view has to be equal to the specified one.");
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::extent::traits::GetExtent
                    {
                        BOOST_REQUIRE_EQUAL(
                            extent,
                            alpaka::extent::getExtentVec(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::mem::view::traits::GetPitchBytes
                    {
                        // The pitches have to be at least as large as the values we calculate here.
                        auto pitchMinimum(alpaka::vec::Vec<alpaka::dim::DimInt<TDim::value + 1u>, TSize>::ones());
                        // Initialize the pitch between two elements of the X dimension ...
                        pitchMinimum[TDim::value] = sizeof(TElem);
                        // ... and fill all the other dimensions.
                        for(TSize i = TDim::value; i > static_cast<TSize>(0u); --i)
                        {
                            pitchMinimum[i-1] = extent[i-1] * pitchMinimum[i];
                        }

                        auto const pitchView(alpaka::mem::view::getPitchBytesVec(view));

                        for(TSize i = TDim::value; i > static_cast<TSize>(0u); --i)
                        {
                            BOOST_REQUIRE_GE(
                                pitchView[i-1],
                                pitchMinimum[i-1]);
                        }
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::mem::view::traits::GetPtrNative
                    {
                        TElem const * const invalidPtr(nullptr);
                        BOOST_REQUIRE_NE(
                            invalidPtr,
                            alpaka::mem::view::getPtrNative(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::offset::traits::GetOffset
                    {
                        BOOST_REQUIRE_EQUAL(
                            offset,
                            alpaka::offset::getOffsetVec(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::size::traits::SizeType
                    {
                        static_assert(
                            std::is_same<alpaka::size::Size<TView>, TSize>::value,
                            "The size type of the view has to be equal to the specified one.");
                    }
                }
            }
        }
    }
}
