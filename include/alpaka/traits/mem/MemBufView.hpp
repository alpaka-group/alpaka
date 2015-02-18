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

#include <alpaka/traits/mem/MemBuf.hpp>     // MemAlloc

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
                typename TMemBuf,
                typename TSfinae = void>
            struct MemBufViewType;

            /*//#############################################################################
            //! The memory buffer view creation type trait.
            //#############################################################################
            template<
                typename TMemBuf,
                typename TExtents,
                typename TOffsets,
                typename TSfinae = void>
            struct CreateMemBufView;*/
        }
    }

    namespace mem
    {
        //#############################################################################
        //! The memory buffer view type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TMemBuf>
        using MemBufViewT = typename traits::mem::MemBufViewType<TMemBuf>::type;

        /*//-----------------------------------------------------------------------------
        //! Constructor.
        //! \param memBuf This can be either a memory buffer base or a memory buffer view itself.
        //! \param offsetsElements The offsets in elements.
        //! \param extentsElements The extents in elements.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf,
            typename TExtents,
            typename TOffsets>
        createMemBufView(
            TMemBuf const & memBuf,
            TExtents const & extentsElements,
            TOffsets const & relativeOffsetsElements)
            -> decltype(traits::mem::CreateMemBufView<TMemBuf, TExtents, TOffsets>::createMemBufView(std::declval<TMemBuf>(), std::declval<TExtents>(), std::declval<TOffsets>()))
        {
            traits::mem::CreateMemBufView<TMemBuf, TExtents, TOffsets>::createMemBufView(
                memBuf,
                extentsElements,
                relativeOffsetsElements)
        }*/
    }
}
