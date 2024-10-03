/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::interface
{
    //! Tag used in class inheritance hierarchies that describes that a specific interface (TInterface)
    //! is implemented by the given base class (TBase).
    template<typename TInterface, typename TBase>
    struct Implements
    {
    };

    //! Checks whether the interface is implemented by the given class
    template<typename TInterface, typename TDerived>
    struct ImplementsInterface
    {
        template<typename TBase>
        static auto implements(Implements<TInterface, TBase>&) -> std::true_type;
        static auto implements(...) -> std::false_type;

        static constexpr auto value = decltype(implements(std::declval<TDerived&>()))::value;
    };

    namespace detail
    {
        //! Returns the type that implements the given interface in the inheritance hierarchy.
        template<typename TInterface, typename TDerived, typename Sfinae = void>
        struct ImplementationBaseType;

        //! Base case for types that do not inherit from "Implements<TInterface, ...>" is the type itself.
        template<typename TInterface, typename TDerived>
        struct ImplementationBaseType<
            TInterface,
            TDerived,
            std::enable_if_t<!ImplementsInterface<TInterface, TDerived>::value>>
        {
            using type = TDerived;
        };

        //! For types that inherit from "Implements<TInterface, ...>" it finds the base class (TBase) which
        //! implements the interface.
        template<typename TInterface, typename TDerived>
        struct ImplementationBaseType<
            TInterface,
            TDerived,
            std::enable_if_t<ImplementsInterface<TInterface, TDerived>::value>>
        {
            template<typename TBase>
            static auto implementer(Implements<TInterface, TBase>&) -> TBase;

            using type = decltype(implementer(std::declval<TDerived&>()));

            static_assert(
                std::is_base_of_v<type, TDerived>,
                "The type implementing the interface has to be a publicly accessible base class!");
        };
    } // namespace detail

    //! Returns the type that implements the given interface in the inheritance hierarchy.
    template<typename TInterface, typename TDerived>
    using ImplementationBase = typename detail::ImplementationBaseType<TInterface, TDerived>::type;
} // namespace alpaka::interface
