/**
* Copyright 2014 Benjamin Worpitz
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

#include <alpaka/interfaces/WorkSize.hpp>   // alpaka::IWorkSize

#include <alpaka/core/Vec.hpp>              // alpaka::vec

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The description of the work size.
            //! This class stores the sizes as members.
            //#############################################################################
            class WorkSizeHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC WorkSizeHost() = default;
                //-----------------------------------------------------------------------------
                //! Constructor from values.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC explicit WorkSizeHost(vec<3u> const & v3uiSizeGridBlocks, vec<3u> const & v3uiSizeBlockKernels) :
                    m_v3uiSizeGridBlocks(v3uiSizeGridBlocks),
                    m_v3uiSizeBlockKernels(v3uiSizeBlockKernels)
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC WorkSizeHost(WorkSizeHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC WorkSizeHost(WorkSizeHost &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC WorkSizeHost & operator=(WorkSizeHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC ~WorkSizeHost() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC vec<3u> getSizeGridBlocks() const
                {
                    return m_v3uiSizeGridBlocks;
                }
                //-----------------------------------------------------------------------------
                //! \return The block dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC vec<3u> getSizeBlockKernels() const
                {
                    return m_v3uiSizeBlockKernels;
                }

            private:
                vec<3u> m_v3uiSizeGridBlocks;
                vec<3u> m_v3uiSizeBlockKernels;
            };
        }
    }

    //#############################################################################
    //! A basic class storing the work to be used in user code.
    //#############################################################################
    using WorkSize = IWorkSize<host::detail::WorkSizeHost>;
}
