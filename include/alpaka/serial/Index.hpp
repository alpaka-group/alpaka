/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/interfaces/Index.hpp>  // IIndex

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            //#############################################################################
            //! This serial accelerator index provider.
            //#############################################################################
            class IndexSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexSerial(
                    vec<3u> const & v3uiGridBlockIdx,
                    vec<3u> const & v3uiBlockKernelIdx) :
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx),
                    m_v3uiBlockKernelIdx(v3uiBlockKernelIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexSerial(IndexSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexSerial(IndexSerial &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexSerial & operator=(IndexSerial const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~IndexSerial() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxBlockKernel() const
                {
                    return m_v3uiBlockKernelIdx;
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                vec<3u> const & m_v3uiGridBlockIdx;
                vec<3u> const & m_v3uiBlockKernelIdx;
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexSerial>;
        }
    }
}
