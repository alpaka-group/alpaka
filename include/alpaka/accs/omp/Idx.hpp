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

#include <alpaka/accs/omp/Common.hpp>

#include <alpaka/traits/Idx.hpp>        // idx::GetIdx

#include <alpaka/core/MapIdx.hpp>       // mapIdx

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace alpaka
{
    namespace accs
    {
        namespace omp
        {
            namespace detail
            {
                //#############################################################################
                //! This OpenMP accelerator index provider.
                //#############################################################################
                template<
                    typename TDim>
                class IdxOmp
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxOmp(
                        Vec<TDim> const & vuiGridBlockIdx) :
                        m_vuiGridBlockIdx(vuiGridBlockIdx)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxOmp(IdxOmp const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxOmp(IdxOmp &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxOmp const &) -> IdxOmp & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxOmp() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread(TWorkDiv const & workDiv) const
                    -> Vec<TDim>
                    {
                        // We assume that the thread id is positive.
                        assert(::omp_get_thread_num()>=0);
                        // \TODO: Would it be faster to precompute the index and cache it inside an array?
                        return mapIdx<TDim::value>(
                            Vec1<>(static_cast<Vec1<>::Val>(::omp_get_thread_num())),
                            alpaka::workdiv::getWorkDiv<Block, Threads>(workDiv));
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The block index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxGridBlock() const
                    -> Vec<TDim>
                    {
                        return m_vuiGridBlockIdx;
                    }

                private:
                    Vec<TDim> const & m_vuiGridBlockIdx; //!< The index of the currently executed block.
                };
            }
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The OpenMP accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::omp::detail::IdxOmp<TDim>>
            {
                using type = TDim;
            };
        }

        namespace idx
        {
            //#############################################################################
            //! The OpenMP accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::omp::detail::IdxOmp<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::omp::detail::IdxOmp<TDim> const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    return index.getIdxBlockThread(workDiv);
                }
            };
            //#############################################################################
            //! The OpenMP accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::omp::detail::IdxOmp<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::omp::detail::IdxOmp<TDim> const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    boost::ignore_unused(workDiv);
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
