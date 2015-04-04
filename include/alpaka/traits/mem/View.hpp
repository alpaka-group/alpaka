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

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The memory buffer view type trait.
            //#############################################################################
            template<
                typename TBuf,
                typename TSfinae = void>
            struct ViewType;

            //#############################################################################
            //! The memory buffer view creation type trait.
            //#############################################################################
            /*template<
                typename TBuf,
                typename TExtents,
                typename TOffsets,
                typename TSfinae = void>
            struct CreateView;*/
        }
    }

    namespace mem
    {
        //#############################################################################
        //! The memory buffer view type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TBuf>
        using ViewT = typename traits::mem::ViewType<TBuf>::type;

        //-----------------------------------------------------------------------------
        //! Constructor.
        //! \param buf This can be either a memory buffer base or a memory buffer view itself.
        //! \param offsetsElements The offsets in elements.
        //! \param extentsElements The extents in elements.
        //-----------------------------------------------------------------------------
        /*template<
            typename TBuf,
            typename TExtents,
            typename TOffsets>
        createBufView(
            TBuf const & buf,
            TExtents const & extentsElements,
            TOffsets const & relativeOffsetsElements)
        -> decltype(traits::mem::CreateView<TBuf, TExtents, TOffsets>::createBufView(std::declval<TBuf const &>(), std::declval<TExtents const &>(), std::declval<TOffsets const &>()))
        {
            traits::mem::CreateView<
                TBuf,
                TExtents,
                TOffsets>
            ::createBufView(
                buf,
                extentsElements,
                relativeOffsetsElements)
        }*/
    }
}
