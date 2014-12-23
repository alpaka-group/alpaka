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

#include <alpaka/interfaces/IAcc.hpp> // alpaka::IAcc

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Extensions to the standard library.
        //#############################################################################
        namespace std_extension
        {
            // This could be replaced with c++14 std::integer_sequence if we raise the minimum.
            template<class T, T... TVals>
            struct integer_sequence
            {
                static_assert(std::is_integral<T>::value, "integer_sequence<T, I...> requires T to be an integral type.");

                typedef integer_sequence<T, TVals...> type;
                typedef T value_type;

                static std::size_t size() noexcept
                {
                    return (sizeof...(TVals));
                }
            };

            template<bool TbNegative, bool TbZero, class TIntCon, class TIntSeq>
            struct make_integer_sequence_helper
            {
                static_assert(!TbNegative, "make_integer_sequence<T, N> requires N to be non-negative.");
            };

            template<class T, T... TVals>
            struct make_integer_sequence_helper<false, true, std::integral_constant<T, 0>, integer_sequence<T, TVals...> > :
                integer_sequence<T, TVals...>
            {};

            template<class T, T TIdx, T... TVals>
            struct make_integer_sequence_helper<false, false, std::integral_constant<T, TIdx>, integer_sequence<T, TVals...> > :
                make_integer_sequence_helper<false, TIdx == 1, std::integral_constant<T, TIdx - 1>, integer_sequence<T, TIdx - 1, TVals...> >
            {};

            template<class T, T TSize>
            using make_integer_sequence = typename make_integer_sequence_helper<(TSize < 0), (TSize == 0), std::integral_constant<T, TSize>, integer_sequence<T> >::type;

            template<std::size_t ... TVals>
            using index_sequence = integer_sequence<std::size_t, TVals...>;

            template<std::size_t TuiSize>
            using make_index_sequence = make_integer_sequence<std::size_t, TuiSize>;

            template<typename... Ts>
            using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
        }

        //#############################################################################
        //! The executor for an accelerated serial kernel.
        //#############################################################################
        template<typename TKernelExecutor, typename... TKernelConstrArgs>
        class KernelExecutorExtent
        {
        public:
            using TAcc = typename TKernelExecutor::TAcc;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent(TKernelConstrArgs && ... args) :
                m_tupleKernelConstrArgs(std::forward<TKernelConstrArgs>(args)...)
            {}
            //-----------------------------------------------------------------------------
            //! Copy-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent(KernelExecutorExtent const &) = default;
            //-----------------------------------------------------------------------------
            //! Move-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent(KernelExecutorExtent &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy-assignment.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent & operator=(KernelExecutorExtent const &) = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~KernelExecutorExtent() noexcept = default;

            //-----------------------------------------------------------------------------
            //! \return An KernelExecutor with the given extents.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST TKernelExecutor operator()(IWorkExtent<TWorkExtent> const & workExtent) const
            {
                return createKernelExecutor(workExtent, TKernelConstrArgsIndexSequence());
            }
            //-----------------------------------------------------------------------------
            //! \return An KernelExecutor with the given extents.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST TKernelExecutor operator()(vec<3u> const & v3uiGridBlocksExtent, vec<3u> const & v3uiBlockKernelsExtent) const
            {
                return createKernelExecutor(WorkExtent(v3uiGridBlocksExtent, v3uiBlockKernelsExtent), TKernelConstrArgsIndexSequence());
            }

        private:
            //-----------------------------------------------------------------------------
            //! \return An KernelExecutor with the given extents.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent, std::size_t ... TIndices>
            ALPAKA_FCT_HOST TKernelExecutor createKernelExecutor(IWorkExtent<TWorkExtent> const & workExtent, std_extension::index_sequence<TIndices...>) const
            {
                if(workExtent.template getExtent<Grid, Blocks, Linear>() == 0)
                {
                    throw std::runtime_error("The workExtent grid blocks extent is not allowed to be zero in any dimension!");
                }
                if(workExtent.template getExtent<Block, Kernels, Linear>() == 0)
                {
                    throw std::runtime_error("The workExtent block kernels extent is not allowed to be zero in any dimension!");
                }

                return TKernelExecutor(workExtent, std::get<TIndices>(std::forward<TKernelConstrArgs>(m_tupleKernelConstrArgs))...);
            }

        private:
            std::tuple<TKernelConstrArgs...> m_tupleKernelConstrArgs;
            using TKernelConstrArgsIndexSequence = std_extension::make_index_sequence<sizeof...(TKernelConstrArgs)>;
        };

        //#############################################################################
        //! The kernel executor builder.
        //#############################################################################
        template<typename TAcc, typename TKernel, typename... TKernelConstrArgs>
        class KernelExecCreator;
    }

    //#############################################################################
    //! Builds a kernel executor.
    //!
    //! Requirements for type TKernel:
    //! The kernel type has to be inherited from 'alpaka::IAcc<boost::mpl::_1>'directly.
    //! All template parameters have to be types. No value parameters are allowed. Use boost::mpl::int_ or similar to use values.
    //#############################################################################
    template<typename TAcc, typename TKernel, typename... TKernelConstrArgs>
    auto createKernelExecutor(TKernelConstrArgs && ... args)
        -> typename std::result_of<detail::KernelExecCreator<TAcc, TKernel, TKernelConstrArgs...>(TKernelConstrArgs...)>::type
    {
        static_assert(std::is_base_of<IAcc<boost::mpl::_1>, TKernel>::value, "The TKernel for createKernelExecutor has to inherit from IAcc<boost::mpl::_1>!");

        // Use the specialized KernelExecCreator for the given accelerator.
        return detail::KernelExecCreator<TAcc, TKernel, TKernelConstrArgs...>()(std::forward<TKernelConstrArgs>(args)...);
    }
}
