/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/concepts/api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/tag.hpp"

#include <type_traits>

/** @brief alpaka'S function interface
 *
 * This file defines the interface for registering, dispatching and calling function overloads specialize for device
 * specifications. A device specification consists of an alpaka API and device kind. These functions can be dispatched
 * to third-party libraries (e.g. cuBLAS) and can be used in alpaka onHost or onAcc. The function interface of alpaka
 * provides a way to work natively with alpaka objects while being able to use third party interfaces for functionality
 * not provided in alpaka or in cases where the vendor implementation provides better performance. For each exposed
 * function you can provide a fallback to an alpaka implementation for a device specification or a device specification
 * independent generic implementation. This keeps your code base portable even if you can not dispatch to a third
 * party/vendor library and avoids preprocessor macros around function calls. The preprocessor macro ALPAKA_FN_SYMBOL()
 * should be used to declare a function symbol.
 * A function symbol follows all requirements to be used as kernel within alpaka.
 *
 * The main components of the interface are:
 * - `alpaka::fn::Fn`: The function symbol baseclass that can be used to register, dispatch and call third party
 * functions.
 * - `alpakaFnRegister`: A function template that can be specialized to register a function overload for a
 * device specification. This is optional and only required if Registration::enforced is set to the function
 * symbol. It allows the usage of isRegistered() function to check if a third party function overload is defined.
 * - `alpakaFnDispatch`: A function template that can be specialized to dispatch a function symbol to a third party
 * function depending on the device specification.
 *
 * For an example of how to use the interface see example/vendorApi.
 */
namespace alpaka::fn
{
    namespace api
    {
        /** @brief Api tag for alpaka.
         *
         * @warning This api should be used together with alpaka's function interface, it is not compatible with other
         * alpaka interfaces where api's are required.
         */
        struct Alpaka : detail::ApiBase
        {
            using element_type = Alpaka;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Alpaka>);
            }

            static std::string getName()
            {
                return "Alpaka";
            }
        };

        constexpr auto alpaka = Alpaka{};
    } // namespace api

    /** @brief Fallback policy for function calls.
     *
     * This enum defines the fallback policy for function calls. It is used as a template parameter in
     * `alpaka::fn::Fn` or ALPAKA_FN_SYMBOL to specify the fallback behavior if no vendor function overload is defined
     * for the given device specification.
     */
    // public enum, the underlying type is part of the interface
    // NOLINTNEXTLINE(performance-enum-size)
    enum class Fallback : int
    {
        /** The generic implementation is called if no other overload fits.
         *
         * Should be used to ensure portability between different heterogeneous APIs.
         */
        toGeneric = 1,
        /** The alpaka implementation is called if no other overload fits.
         *
         * Should be used to ensure portability between different heterogeneous APIs.
         */
        toAlpaka = 2,
        /** No fallback is performed in case no overload is fitting.
         *
         * Should be used if you want to ensure that a third party function overload is guaranteed to be called.
         */
        none = 3
    };

    /** @brief Policy to control if a function symbol must be registered.
     */
    // public enum, the underlying type is part of the interface
    // NOLINTNEXTLINE(performance-enum-size)
    enum class Registration : int
    {
        /** The isRegistered() function will always return true. This can be used to skip the registration of the
         * function symbol via alpakaFnRegister().
         */
        alwaysTrue = 1,
        /** It is required to define alpakaFnRegister() for a function symbol. isRegistered() can be called to check if
         * a vendor function overload is registered for the given device specification.
         */
        enforced = 2,
        /** The isRegistered() function is not available and no registration of the vendor function overloads is
         * required. This can be used if you do not want to use the isRegistered() function and do not want to require
         * the definition of alpakaFnRegister() for a function symbol.
         */
        none = 3
    };

    namespace concepts
    {
        /** @brief Concept to check if a function symbol can be called.
         *
         * This concept checks if the alpakaFnDispatch() can be called with the given function symbol (if
         * Fallback::toGeneric) or function symbol device specification. It is used to check if a function dispatch is
         * defined for the given device specification or function symbol and if it can be called with the given
         * arguments.
         */
        template<typename T_FnSpec, typename... T_Args>
        concept DispatchedFnInvocable
            = requires(T_FnSpec fnSpec, T_Args&&... args) { alpakaFnDispatch(fnSpec, std::forward<T_Args>(args)...); };

        /** @brief Concept to check if a function symbol is registered.
         *
         * This concept checks if the alpakaFnRegister() can be called with the given function symbol or
         * function symbol device specification. It is used to check if a vendor function overload is defined for the
         * given device specification without taking any function arguments into account.
         */
        template<typename T_FnSpec>
        concept FnRegistered = requires(T_FnSpec fnSpec) { alpakaFnRegister(fnSpec); };
    } // namespace concepts

    /** @brief Base class for function symbols.
     *
     * @tparam T_FnClass The function symbol to register, dispatch and call. The class should be trivially
     * constructable. By using the static call() function or the operator()
     * @tparam T_fallbackPolicy The fallback policy if no vendor function overload is defined for the given device
     * specification. If set to Fallback::toAlpaka the alpaka implementation is called if no other overload fits. If
     * set to Fallback::none no fallback is performed and a static assert is triggered if no function overload is
     * defined for the given device specification.
     * @tparam T_registrationPolicy If set to Registration::enforced the isRegistered() can be called, and it is
     * required to define alpakaFnRegister() for on T_FnClass. If set to Registration::none the isRegistered()
     * function is not available and no registration of the vendor function overloads is required. If set to
     * Registration::alwaysTrue isRegistered() will always return true. This can be used to skip the registration
     * of the function symbol.
     */
    template<
        typename T_FnClass,
        Fallback T_fallbackPolicy = Fallback::toGeneric,
        Registration T_registrationPolicy = Registration::none>
    // symbols derived without an own constructor, e.g. by ALPAKA_FN_SYMBOL, are aggregates;
    // a private constructor would break their initialization
    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    struct Fn
    {
        /** Get the function specification.
         *
         * @return the function specification for the given entity.
         *
         * @{
         */
        static constexpr auto spec(alpaka::concepts::DeviceSpec auto const& any)
        {
            return spec(getApi(any), getDeviceKind(any));
        }

        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
        static constexpr auto spec(T_Api api, T_DeviceKind deviceKind)
        {
            alpaka::unused(api, deviceKind);
            return typename T_FnClass::template Spec<T_Api, T_DeviceKind>{};
        }

        /** @} */

        /** Checks if a function overload is registered for the given device specification.
         *
         * You can use the result to optionally call the function overload and disable at compile time code sections
         * similar to C++ preprocessor guards.
         * @code
         * ALPAKA_FN_SYMBOL(Foo,alpaka::fn::Fallback::none, alpaka::fn::Registration::enforced);
         *
         * void alpakaFnRegister(Foo::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>)
         * {
         * }
         *
         * if constexpr (Foo::isRegistered(queue))
         * {
         *     // more code
         *     Foo::call(queue,args ...);
         *     // more code
         * }
         * @endcode
         *
         * @param any any type which is usable with alpaka::getApi() and alpaka::getDeviceKind()
         * @return true if the function overload T_FnClass is registered else false.
         * It does not try to check if a fallback overload is dispatchable, to check fallback registrations use
         * hasRegisteredFallback(alpaka::concepts::DeviceSpec auto const& any).
         */
        static constexpr bool isRegistered(alpaka::concepts::DeviceSpec auto const& any)
            requires(T_registrationPolicy != Registration::none)
        {
            return T_registrationPolicy == Registration::alwaysTrue
                   || concepts::FnRegistered<ALPAKA_TYPEOF(spec(any))>;
        }

        /** Checks if the function overload fallback is registered.
         *
         * Similar to isRegistered(alpaka::concepts::DeviceSpec auto const& any) but it checks for the fallback
         * overload only.
         *
         * @param any any type which is usable with alpaka::getApi() and alpaka::getDeviceKind()
         * @return true if the function overload fallback for T_FnClass is registered else false.
         */
        static constexpr bool hasRegisteredFallback(alpaka::concepts::DeviceSpec auto const& any)
            requires(T_registrationPolicy != Registration::none)
        {
            constexpr bool isFallbackAllowed = T_fallbackPolicy != Fallback::none;
            constexpr bool hasAlpakaFallback
                = ((T_fallbackPolicy == Fallback::toAlpaka)
                   && concepts::FnRegistered<ALPAKA_TYPEOF(spec(api::Alpaka{}, getDeviceKind(any)))>);
            constexpr bool hasGenericFallback
                = ((T_fallbackPolicy == Fallback::toGeneric) && concepts::FnRegistered<T_FnClass>);
            return T_registrationPolicy == Registration::alwaysTrue
                   || (isFallbackAllowed && (hasAlpakaFallback || hasGenericFallback));
        }

        /** Call function overload if defined for the given device specification. */
        template<alpaka::concepts::DeviceSpec T_Any, typename... T_Args>
        requires concepts::DispatchedFnInvocable<ALPAKA_TYPEOF(spec(std::declval<T_Any>())), T_Any, T_Args...>
        constexpr decltype(auto) operator()(T_Any&& any, T_Args&&... args) const
        {
            static_assert(
                T_registrationPolicy != Registration::enforced || concepts::FnRegistered<ALPAKA_TYPEOF(spec(any))>,
                "Function dispatch for the given function symbol, API and device kind is not registered.");
            return alpakaFnDispatch(spec(any), std::forward<T_Any>(any), std::forward<T_Args>(args)...);
        }

        /** Fallback operator() to alpaka implementation if the function is not dispatchable for the given device
         * specification.
         *
         * This operator() is only enabled if T_fallbackPolicy is set to toAlpaka and function is
         * dispatchable for the given device specification.
         */
        template<alpaka::concepts::DeviceSpec T_Any, typename... T_Args>
        requires(
            !concepts::DispatchedFnInvocable<ALPAKA_TYPEOF(spec(std::declval<T_Any>())), T_Any, T_Args...>
            && (T_fallbackPolicy == Fallback::toAlpaka))
        constexpr decltype(auto) operator()(T_Any&& any, T_Args&&... args) const
        {
            static_assert(
                T_registrationPolicy != Registration::enforced
                    || concepts::FnRegistered<ALPAKA_TYPEOF(spec(api::Alpaka{}, getDeviceKind(any)))>,
                "Function for the given function group, device kind the api fn::api::alpaka is not registered.");
            return alpakaFnDispatch(
                spec(api::Alpaka{}, getDeviceKind(any)),
                std::forward<T_Any>(any),
                std::forward<T_Args>(args)...);
        }

        /** Fallback operator() to generic function if not dispatchable for the given device
         * specification.
         *
         * This operator() is only enabled if T_fallbackPolicy is set toGeneric and function is
         * dispatchable without a device specification.
         */
        template<alpaka::concepts::DeviceSpec T_Any, typename... T_Args>
        requires(
            // no dispatch with device specification
            !concepts::DispatchedFnInvocable<ALPAKA_TYPEOF(spec(std::declval<T_Any>())), T_Any, T_Args...> &&
            // generic function dispatchable
            concepts::DispatchedFnInvocable<T_FnClass, T_Any, T_Args...> && (T_fallbackPolicy == Fallback::toGeneric))
        constexpr decltype(auto) operator()(T_Any&& any, T_Args&&... args) const
        {
            static_assert(
                T_registrationPolicy != Registration::enforced || concepts::FnRegistered<T_FnClass>,
                "Function dispatch for the given function symbol, is not registered.");
            return alpakaFnDispatch(T_FnClass{}, std::forward<T_Any>(any), std::forward<T_Args>(args)...);
        }

        /** Call the function overload for the given device specification.
         *
         * See the call operator().
         * @attention call() is a static function where the call operator required an instance of this class.
         */
        static constexpr decltype(auto) call(alpaka::concepts::DeviceSpec auto&& any, auto&&... args)
        {
            static_assert(
                std::is_trivially_constructible_v<T_FnClass>,
                "Function class must be trivially constructible to use call().");
            return T_FnClass{}(ALPAKA_FORWARD(any), ALPAKA_FORWARD(args)...);
        }
    };
} // namespace alpaka::fn

/** @brief Define a function symbol class for alpaka's function interface
 *
 * @param fnName Name of the function symbol. This can be used to call the function overloads with the static
 * call() function without having to create an instance or with the operator().
 * @param optional_fallback The fallback policy if no vendor function overload is defined for the given device
 * specification. If set to Fallback::toAlpaka the generic alpaka implementation is called if no other overload is
 * fitting. If set to Fallback::toGeneric the overload with the function symbol as first argument is called if no other
 * function device specification is callable. If set to Fallback::none no fallback is performed and a static assert is
 * triggered if no vendor function overload is defined for the given device specification. Default:
 * Fallback::toGeneric.
 * @param optional_registartion If set to Registration::enforced the isRegistered() can be called, and it is
 * required to define alpakaFnRegister() for on T_FnClass. If set to Registration::none the isRegistered()
 * function is not available and no registration of the vendor function overloads is required. If set to
 * Registration::alwaysTrue isRegistered() will always return true. This can be used to skip the registration of
 * the function symbol. Default: Registration::none.
 *
 * @code
 * ALPAKA_FN_SYMBOL(Transform, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::enforced);
 * ALPAKA_FN_SYMBOL(TransformWithFallback, alpaka::fn::Fallback::toAlpaka);
 * ALPAKA_FN_SYMBOL(TransformWithFallbackAndRegistration, alpaka::fn::Fallback::toGeneric,
 * alpaka::fn::Registration::enforced);
 * @endcode
 */
#define ALPAKA_FN_SYMBOL(fnName, ...)                                                                                 \
    struct fnName : alpaka::fn::Fn<fnName __VA_OPT__(, __VA_ARGS__)>                                                  \
    {                                                                                                                 \
        /** Function specification for a given device specification.                                                  \
         *                                                                                                            \
         * This struct should be specialized for each device specification combination where a vendor                 \
         * function overload is defined. The specialization should be empty and only used as a tag to                 \
         * identify the function overload.                                                                            \
         */                                                                                                           \
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>                              \
        struct Spec                                                                                                   \
        {                                                                                                             \
        };                                                                                                            \
    };                                                                                                                \
    static_assert(true)
