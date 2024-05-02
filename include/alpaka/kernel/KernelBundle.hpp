/* Copyright 2022 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber, Mehmet
 * Yusufoglu SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/DemangleTypeNames.hpp>
#include <alpaka/core/RemoveRestrict.hpp>

#include <tuple>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //! Check if a type used as kernel argument is trivially copyable
        //!
        //! \attention In case this trait is specialized for a user type the user should be sure that the result of
        //! calling the copy constructor is equal to use memcpy to duplicate the object. An existing destructor should
        //! be free of side effects.
        //!
        //! It's implementation defined whether the closure type of a lambda is trivially copyable.
        //! Therefor the default implementation is true for trivially copyable or empty (stateless) types.
        //!
        //! @tparam T type to check
        //! @{
        template<typename T, typename = void>
        struct IsKernelArgumentTriviallyCopyable
            : std::bool_constant<std::is_empty_v<T> || std::is_trivially_copyable_v<T>>
        {
        };

        template<typename T>
        inline constexpr bool isKernelArgumentTriviallyCopyable = IsKernelArgumentTriviallyCopyable<T>::value;

        // asserts that T is trivially copyable. We put this in a separate function so we can see which T would fail
        // the test, when called from a fold expression.
        template<typename T>
        inline void assertKernelArgIsTriviallyCopyable()
        {
            static_assert(isKernelArgumentTriviallyCopyable<T>, "The kernel argument T must be trivially copyable!");
        }

    } // namespace detail

    //! \brief The class used to bind kernel function object and arguments together. Once an instance of this class is
    //! created, arguments are not needed to be separately given to functions who need kernel function and arguments.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object invocation argument types as a parameter pack.
    template<typename TKernelFn, typename... TArgs>
    class KernelBundle
    {
    public:
        //! The function object type
        using KernelFn = TKernelFn;
        //! Tuple type to encapsulate kernel function argument types and argument values
        using ArgTuple = std::tuple<remove_restrict_t<std::decay_t<TArgs>>...>;

        // Constructor
        KernelBundle(KernelFn kernelFn, TArgs&&... args)
            : m_kernelFn(std::move(kernelFn))
            , m_args(std::forward<TArgs>(args)...)
        {
#if BOOST_COMP_NVCC
            static_assert(
                std::is_trivially_copyable_v<TKernelFn> || __nv_is_extended_device_lambda_closure_type(TKernelFn)
                    || __nv_is_extended_host_device_lambda_closure_type(TKernelFn),
                "Kernels must be trivially copyable or an extended CUDA lambda expression!");
#else
            static_assert(std::is_trivially_copyable_v<TKernelFn>, "Kernels must be trivially copyable!");
#endif
            (detail::assertKernelArgIsTriviallyCopyable<std::decay_t<TArgs>>(), ...);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << ", kernelFnObj: " << core::demangled<decltype(m_kernelFn)> << std::endl;
#endif
        }

        KernelFn m_kernelFn;
        ArgTuple m_args; // Store the argument types without const and reference
    };

    //! \brief User defined deduction guide with trailing return type. For CTAD during the construction.
    //! \tparam TKernelFn The kernel function object type.
    //! \tparam TArgs Kernel function object argument types as a parameter pack.
    //! \param kernelFn The kernel object
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation" // clang does not support the syntax for variadic template
                                                       // arguments "args,...". Ignore the error.
#endif
    //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    //! \return Kernel function bundle. An instance of KernelBundle which consists the kernel function object and its
    //! arguments.
    template<typename TKernelFn, typename... TArgs>
    ALPAKA_FN_HOST KernelBundle(TKernelFn, TArgs&&...) -> KernelBundle<TKernelFn, TArgs...>;
} // namespace alpaka
