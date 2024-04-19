/* Copyright 2022 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/core/RemoveRestrict.hpp>

#include <tuple>
#include <type_traits>

namespace alpaka
{
    //! \brief The class used to bind kernel function object and arguments together. Once an instance of this class is
    //! created, arguments are not needed to be separately given to functions who need kernel function and arguments.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object invocation argument types as a parameter pack.
    template<typename TKernelFn, typename... TArgs>
    class KernelBundle
    {
    public:
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation" // clang does not support the syntax for variadic template
                                                       // arguments "args,...". Ignore the error.
#endif
        //! \param kernelFn The kernel function-object
        //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
        KernelBundle(TKernelFn const& kernelFn, TArgs&&... args)
            : m_kernelFn(kernelFn)
            , m_args(std::forward<TArgs>(args)...)
        {
        }

        using KernelFn = TKernelFn;
        using ArgTuple = std::tuple<remove_restrict_t<std::decay_t<TArgs>>...>;

        KernelFn m_kernelFn;
        ArgTuple m_args;
    };

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation" // clang does not support the syntax for variadic template
                                                       // arguments "args,...". Ignore the error.
#endif
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    //! \param kernelFn The kernel object
    //! \param args,... The kernel invocation arguments.
    //! \return Kernel function bundle. An instance of KernelBundle which consists the kernel function object and its
    //! arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    template<typename TKernelFn, typename... TArgs>
    inline auto makeKernelBundle(TKernelFn const& kernelFn, TArgs&&... args) -> KernelBundle<TKernelFn, TArgs...>
    {
        return KernelBundle<TKernelFn, TArgs...>(kernelFn, std::forward<TArgs>(args)...);
    }

    // additional deduction guide
    template<typename TKernelFn, typename... TArgs>
    KernelBundle(TKernelFn const& kernelFn, TArgs&&... args) -> KernelBundle<TKernelFn, TArgs...>;

} // namespace alpaka
