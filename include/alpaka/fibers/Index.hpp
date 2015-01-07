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

#include <alpaka/fibers/Common.hpp>

#include <alpaka/interfaces/Index.hpp>  // IIndex

#include <map>                          // std::map

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            using TFiberIdToIndex = std::map<boost::fibers::fiber::id, vec<3u>>;
            //#############################################################################
            //! This fibers accelerator index provider.
            //#############################################################################
            class IndexFibers
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexFibers(
                    TFiberIdToIndex const & mFibersToIndices,
                    vec<3u> const & v3uiGridBlockIdx) :
                    m_mFibersToIndices(mFibersToIndices),
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexFibers(IndexFibers const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexFibers(IndexFibers &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexFibers & operator=(IndexFibers const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~IndexFibers() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxBlockKernel() const
                {
                    auto const idFiber(boost::this_fiber::get_id());
                    auto const itFind(m_mFibersToIndices.find(idFiber));
                    assert(itFind != m_mFibersToIndices.end());

                    return itFind->second;
                }
                //-----------------------------------------------------------------------------
                //! \return The index of the block of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                TFiberIdToIndex const & m_mFibersToIndices;     //!< The mapping of fibers id's to fibers indices.
                vec<3u> const & m_v3uiGridBlockIdx;             //!< The index of the currently executed block.
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexFibers>;
        }
    }
}
