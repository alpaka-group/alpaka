/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Assert.hpp>

#include <type_traits>


namespace alpaka
{
    namespace core
    {
        /**
         * Scopes are required for BindScope functor and the bindScope function.
         */
        enum class Scope {
            HostOnly,
            DeviceOnly,
            HostDevice,
            Default
        };

        /**
         * BindScope wraps a callable object and executes it only in the given scope.
         *
         * A BindScope object can always be created both on host and on device.
         * The callable is only called when the provided scope matches.
         * The behaviour is undefined, when a method is called outside the specified scope.
         *
         * HIP(HC): BindScope adds missing host or device functions, otherwise HC will complain.
         * Note, that in HIP/HC the host-device annotations belong to the function signature (attributes are applied to the function type).
         *
         * NVCC: BindScope invokes the callable only in the specified scope and the compiler stage,
         * and calls ALPAKA_ASSERT(false) otherwise.
         * A lambda still needs host or device annotation, use ALPAKA_LAMBDA.
         */
        template<typename TFunc, Scope TScope>
        struct BindScope;

        /**
         * Returns BindScope wrapper callable for HIP and CUDA backends.
         *
         * For the other backends it just returns the callable itself.
         */
        template<Scope TScope, typename TFunc>
        ALPAKA_FN_INLINE
#if BOOST_LANG_CUDA || BOOST_LANG_HIP
        auto bindScope( const TFunc callable ) -> BindScope<TFunc, TScope> {
            return BindScope<TFunc, TScope>( callable );
        }
#else
        auto bindScope( const TFunc callable ) -> TFunc {
            // no host-device semantics, so no wrapper needed, just return the callable
            return callable;
        }
#endif



// --- HIP(NVCC) ---
#if BOOST_COMP_NVCC || BOOST_COMP_CLANG_CUDA

        // Default NVCC: enable host + device for callable.
        template<typename TFunc, Scope TScope>
        struct BindScope {
            const TFunc m_callable;

            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

            ALPAKA_NO_HOST_ACC_WARNING
            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }
        };

        // HostOnly NVCC:
        template<typename TFunc>
        struct BindScope<TFunc, Scope::HostOnly> {
            const TFunc m_callable;

            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

#if BOOST_ARCH_PTX
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC
            auto operator()(TArgs&&... args) const -> void { // void, otherwise callable() must be invoked for decltype()
                ALPAKA_ASSERT(false);
                alpaka::ignore_unused(args...);
            }
#else
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }
#endif
        };

        // DeviceOnly NVCC: Lambdas must be annotated with __device__ too (see ALPAKA_LAMBDA)
        template<typename TFunc>
        struct BindScope<TFunc, Scope::DeviceOnly> {
            const TFunc m_callable;

            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

#if !BOOST_ARCH_PTX
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC
            auto operator()(TArgs&&... args) const -> void { // void, otherwise callable() must be invoked for decltype()
                ALPAKA_ASSERT(false);
                alpaka::ignore_unused(args...);
            }
#else
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }
#endif
        };



// --- HIP(HCC) ---
#elif BOOST_COMP_HCC

        // Default HIP/HCC: let HC decide and just call the callable without host-device annotation.
        template<typename TFunc, Scope TScope=Scope::Default>
        struct BindScope {
            const TFunc m_callable;

            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

            template<typename... TArgs>
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }

            ALPAKA_FN_HOST_ACC ~BindScope(){} // destructor must reflect host-device annotation of constructors
        };

        // HostDevice HIP/HCC: explicitly define host+device context
        template<typename TFunc>
        struct BindScope<TFunc, Scope::HostDevice> {
            const TFunc m_callable;

            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }

            ALPAKA_FN_HOST_ACC ~BindScope(){} // destructor must reflect host-device annotation of constructors
        };

        // HostOnly HIP/HCC:
        template<typename TFunc>
        struct BindScope<TFunc, Scope::HostOnly> {
            const TFunc m_callable;

            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

            template<typename... TArgs>
            ALPAKA_FN_HOST
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }
            // do not use decltype for trailing return type, it will break HCC with AMP restricted call in host function
            template<typename T, typename... TArgs>
            __device__
            auto operator()(T&&, TArgs&&... args) const -> void { // void, otherwise callable() must be invoked for decltype()
                ALPAKA_ASSERT(false);
                alpaka::ignore_unused(args...);
            }

            ALPAKA_FN_HOST_ACC ~BindScope(){} // destructor must reflect host-device annotation of constructors
        };

        template<typename TFunc>
        struct BindScope<TFunc, Scope::DeviceOnly> {
            const TFunc m_callable;

            ALPAKA_FN_HOST_ACC
            explicit BindScope(const TFunc callable) : m_callable{callable} {}

            template<typename... TArgs>
            ALPAKA_FN_HOST
            auto operator()(TArgs&&... args) const -> void { // void, otherwise callable() must be invoked for decltype()
                ALPAKA_ASSERT(false);
                alpaka::ignore_unused(args...);
            }
            template<typename... TArgs>
            __device__
            auto operator()(TArgs&&... args) const -> decltype(m_callable(std::forward<TArgs>(args)...)) {
                return m_callable(std::forward<TArgs>(args)...);
            }

            ALPAKA_FN_HOST_ACC ~BindScope(){} // destructor must reflect host-device annotation of constructors
        };
#endif // HCC

    } // core
} // alpaka

// macro definitions to provide shortcuts
#define ALPAKA_FN_SCOPE_HOST alpaka::core::bindScope< alpaka::core::Scope::HostOnly >
#define ALPAKA_FN_SCOPE_DEVICE alpaka::core::bindScope< alpaka::core::Scope::DeviceOnly >
#define ALPAKA_FN_SCOPE_HOST_DEVICE alpaka::core::bindScope< alpaka::core::Scope::HostDevice >
