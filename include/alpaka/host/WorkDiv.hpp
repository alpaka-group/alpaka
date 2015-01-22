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

#include <alpaka/interfaces/WorkDiv.hpp> // alpaka::IWorkDiv

#include <alpaka/core/Vec.hpp>              // alpaka::Vec

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The host accelerators work division.
            //#############################################################################
            class WorkDivHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkDivHost() = default;
                //-----------------------------------------------------------------------------
                //! Constructor from values.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST explicit WorkDivHost(
                    Vec<3u> const & v3uiGridBlocksExtent, 
                    Vec<3u> const & v3uiBlockKernelsExtents) :
                    m_v3uiGridBlocksExtents(v3uiGridBlocksExtent),
                    m_v3uiBlockKernelsExtents(v3uiBlockKernelsExtents)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkDivHost(WorkDivHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkDivHost(WorkDivHost &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST WorkDivHost & operator=(WorkDivHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~WorkDivHost() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid blocks extents of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST Vec<3u> getGridBlocksExtents() const
                {
                    return m_v3uiGridBlocksExtents;
                }
                //-----------------------------------------------------------------------------
                //! \return The block kernels extents of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST Vec<3u> getBlockKernelsExtents() const
                {
                    return m_v3uiBlockKernelsExtents;
                }

            private:
                Vec<3u> m_v3uiGridBlocksExtents;
                Vec<3u> m_v3uiBlockKernelsExtents;
            };
        }
    }

    //#############################################################################
    //! A basic class storing the work division.
    //#############################################################################
    using WorkDiv = IWorkDiv<host::detail::WorkDivHost>;
}
