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

#include <alpaka/interfaces/WorkExtent.hpp> // alpaka::IWorkExtent

#include <alpaka/core/Vec.hpp>              // alpaka::vec

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The host accelerators work extent.
            //#############################################################################
            class WorkExtentHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkExtentHost() = default;
                //-----------------------------------------------------------------------------
                //! Constructor from values.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST explicit WorkExtentHost(vec<3u> const & v3uiGridBlocksExtent, vec<3u> const & v3uiBlockKernelsExtent) :
                    m_v3uiGridBlocksExtent(v3uiGridBlocksExtent),
                    m_v3uiBlockKernelsExtent(v3uiBlockKernelsExtent)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkExtentHost(WorkExtentHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkExtentHost(WorkExtentHost &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkExtentHost & operator=(WorkExtentHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~WorkExtentHost() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getExtentGridBlocks() const
                {
                    return m_v3uiGridBlocksExtent;
                }
                //-----------------------------------------------------------------------------
                //! \return The block dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getExtentBlockKernels() const
                {
                    return m_v3uiBlockKernelsExtent;
                }

            private:
                vec<3u> m_v3uiGridBlocksExtent;
                vec<3u> m_v3uiBlockKernelsExtent;
            };
        }
    }

    //#############################################################################
    //! A basic class storing the work to be used in user code.
    //#############################################################################
    using WorkExtent = IWorkExtent<host::detail::WorkExtentHost>;
}
