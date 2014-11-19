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

namespace alpaka
{
    namespace detail
    {
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
    //! The kernel type has to have at least one template parameters 'typename TAcc = boost::mpl::_1' and has to inherit indirectly from this type publicly via the IAcc interface 'public alpaka::IAcc<TAcc>'.
    //! All template parameters have to be types. No value parameters are allowed. Use boost::mpl::int_ or similar to use values.
    //! TODO: Check these requirements at compile time!
    //#############################################################################
    template<typename TAcc, typename TKernel, typename... TKernelConstrArgs>
    auto createKernelExecutor(TKernelConstrArgs && ... args)
        -> typename std::result_of<detail::KernelExecCreator<TAcc, TKernel, TKernelConstrArgs...>(TKernelConstrArgs...)>::type
    {
        // Use the specialized KernelExecCreator for the given accelerator.
        return detail::KernelExecCreator<TAcc, TKernel, TKernelConstrArgs...>()(std::forward<TKernelConstrArgs>(args)...);
    }
}
