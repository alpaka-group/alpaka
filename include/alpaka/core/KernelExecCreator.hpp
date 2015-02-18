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

#include <alpaka/interfaces/IAcc.hpp>        // IAcc

#include <alpaka/core/WorkDivHelpers.hpp>    // workdiv::isValidWorkDiv
#include <alpaka/core/IntegerSequence.hpp>   // workdiv::isValidWorkDiv

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The executor for an accelerated serial kernel.
        //#############################################################################
        template<
            typename TKernelExecutor, 
            typename... TKernelConstrArgs>
        class KernelExecutorExtent
        {
            using Acc = acc::AccT<TKernelExecutor>;
            using Stream = stream::StreamT<Acc>;
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent(TKernelConstrArgs && ... args) :
                m_tupleKernelConstrArgs(std::forward<TKernelConstrArgs>(args)...)
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent(KernelExecutorExtent const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent(KernelExecutorExtent &&) = default;
#endif
            //-----------------------------------------------------------------------------
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent & operator=(KernelExecutorExtent const &) = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST virtual ~KernelExecutorExtent() noexcept = default;

            //-----------------------------------------------------------------------------
            //! \return An KernelExecutor with the given extents.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST TKernelExecutor operator()(
                TWorkDiv const & workDiv, 
                Stream const & stream) const
            {
                if(!workdiv::isValidWorkDiv<Acc>(workDiv))
                {
                    throw std::runtime_error("The given work division is not supported by the " + acc::getAccName<Acc>() + " accelerator!");
                }

                return createKernelExecutor(
                    workDiv, 
                    stream, 
                    KernelConstrArgsIdxSequence());
            }
            //-----------------------------------------------------------------------------
            //! \return An KernelExecutor with the given extents.
            //-----------------------------------------------------------------------------
            template<
                typename TGridBlocksExtents,
                typename TBlockKernelsExtents>
            ALPAKA_FCT_HOST TKernelExecutor operator()(
                TGridBlocksExtents const & gridBlocksExtent,
                TBlockKernelsExtents const & blockKernelsExtents, 
                Stream const & stream) const
            {
                return this->operator()(
                    workdiv::BasicWorkDiv(
                        Vec<3u>(extent::getWidth(gridBlocksExtent), extent::getHeight(gridBlocksExtent), extent::getDepth(gridBlocksExtent)),
                        Vec<3u>(extent::getWidth(blockKernelsExtents), extent::getHeight(blockKernelsExtents), extent::getDepth(blockKernelsExtents))), 
                    stream);
            }

        private:
            //-----------------------------------------------------------------------------
            //! \return An KernelExecutor with the given extents.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv,
                std::size_t... TIndices>
            ALPAKA_FCT_HOST TKernelExecutor createKernelExecutor(
                TWorkDiv const & workDiv, 
                Stream const & stream,
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
                index_sequence<TIndices...> const &) const
#else
                integer_sequence<std::size_t, TIndices...> const &) const
#endif
            {
                if(workdiv::getWorkDiv<Grid, Blocks, dim::Dim1>(workDiv)[0] == 0u)
                {
                    throw std::runtime_error("The workDiv grid blocks extents is not allowed to be zero in any dimension!");
                }
                if(workdiv::getWorkDiv<Block, Kernels, dim::Dim1>(workDiv)[0] == 0u)
                {
                    throw std::runtime_error("The workDiv block kernels extents is not allowed to be zero in any dimension!");
                }

                return TKernelExecutor(workDiv, stream, std::forward<TKernelConstrArgs>(std::get<TIndices>(m_tupleKernelConstrArgs))...);
            }

        private:
            std::tuple<TKernelConstrArgs...> m_tupleKernelConstrArgs;
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using KernelConstrArgsIdxSequence = typename make_index_sequence<sizeof...(TKernelConstrArgs)>::type;
#else
            using KernelConstrArgsIdxSequence = make_index_sequence<sizeof...(TKernelConstrArgs)>;
#endif
        };

        //#############################################################################
        //! The kernel executor builder.
        //#############################################################################
        template<
            typename TAcc, 
            typename TKernel, 
            typename... TKernelConstrArgs>
        class KernelExecCreator;
    }

    //#############################################################################
    //! Builds a kernel executor.
    //!
    //! Requirements for type TKernel:
    //! The kernel type has to be inherited from 'alpaka::IAcc<boost::mpl::_1>'directly.
    //! All template parameters have to be types. No value parameters are allowed. Use boost::mpl::int_ or similar to use values.
    //#############################################################################
    template<
        typename TAcc, 
        typename TKernel, 
        typename... TKernelConstrArgs>
    auto createKernelExecutor(
        TKernelConstrArgs && ... args)
    -> typename std::result_of<detail::KernelExecCreator<TAcc, TKernel, TKernelConstrArgs...>(TKernelConstrArgs...)>::type
    {
#if (!BOOST_COMP_GNUC) || (BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(5, 0, 0))
        static_assert(std::is_trivially_copyable<TKernel>::value, "The given kernel functor has to fulfill is_trivially_copyable!");
#endif

        // Use the specialized KernelExecCreator for the given accelerator.
        return detail::KernelExecCreator<TAcc, TKernel, TKernelConstrArgs...>()(std::forward<TKernelConstrArgs>(args)...);
    }
}
