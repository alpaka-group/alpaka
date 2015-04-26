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

#include <alpaka/accs/fibers/Common.hpp>

#include <alpaka/traits/Idx.hpp>    // idx::getIdx

#include <map>                      // std::map

namespace alpaka
{
    namespace accs
    {
        namespace fibers
        {
            namespace detail
            {
                using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, Vec3<>>;
                //#############################################################################
                //! This fibers accelerator index provider.
                //#############################################################################
                class IdxFibers
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxFibers(
                        FiberIdToIdxMap const & mFibersToIndices,
                        Vec3<> const & v3uiGridBlockIdx) :
                        m_mFibersToIndices(mFibersToIndices),
                        m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxFibers(IdxFibers const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxFibers(IdxFibers &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxFibers const &) -> IdxFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxFibers() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread() const
                    -> Vec3<>
                    {
                        auto const idFiber(boost::this_fiber::get_id());
                        auto const itFind(m_mFibersToIndices.find(idFiber));
                        assert(itFind != m_mFibersToIndices.end());

                        return itFind->second;
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The index of the block of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxGridBlock() const
                    -> Vec3<>
                    {
                        return m_v3uiGridBlockIdx;
                    }

                private:
                    FiberIdToIdxMap const & m_mFibersToIndices;     //!< The mapping of fibers id's to fibers indices.
                    Vec3<> const & m_v3uiGridBlockIdx;             //!< The index of the currently executed block.
                };
            }
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The fibers accelerator 3D block thread index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                accs::fibers::detail::IdxFibers,
                origin::Block,
                unit::Threads,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::fibers::detail::IdxFibers const & index,
                    TWorkDiv const &)
                -> alpaka::Vec<alpaka::dim::Dim3>
                {
                    return index.getIdxBlockThread();
                }
            };

            //#############################################################################
            //! The fibers accelerator 3D grid block index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                accs::fibers::detail::IdxFibers,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::fibers::detail::IdxFibers const & index,
                    TWorkDiv const &)
                -> alpaka::Vec<alpaka::dim::Dim3>
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
