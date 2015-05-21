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

#include <alpaka/traits/Idx.hpp>        // idx::getIdx

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace alpaka
{
    namespace accs
    {
        namespace serial
        {
            namespace detail
            {
                //#############################################################################
                //! This serial accelerator index provider.
                //#############################################################################
                template<
                    typename TDim>
                class IdxSerial
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxSerial(
                        Vec<TDim> const & vuiGridBlockIdx) :
                        m_vuiGridBlockIdx(vuiGridBlockIdx)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxSerial(IdxSerial const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxSerial(IdxSerial &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxSerial const &) -> IdxSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxSerial() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread() const
                    -> Vec<TDim>
                    {
                        return Vec<TDim>::zeros();
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
                    Vec<TDim> const & m_vuiGridBlockIdx;
                };
            }
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The CPU serial accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::serial::detail::IdxSerial<TDim>>
            {
                using type = TDim;
            };
        }

        namespace idx
        {
            //#############################################################################
            //! The CPU serial accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::serial::detail::IdxSerial<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::serial::detail::IdxSerial<TDim> const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    boost::ignore_unused(workDiv);
                    return index.getIdxBlockThread();
                }
            };

            //#############################################################################
            //! The CPU serial accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::serial::detail::IdxSerial<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::serial::detail::IdxSerial<TDim> const & index,
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
