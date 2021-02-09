/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <CL/sycl.hpp>

#include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The SYCL accelerator atomic ops.
    //
    //  Atomics can used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices
    class AtomicGenericSycl
    {
    public:

        //-----------------------------------------------------------------------------
        AtomicGenericSycl() = default;
        //-----------------------------------------------------------------------------
        AtomicGenericSycl(AtomicGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        AtomicGenericSycl(AtomicGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AtomicGenericSycl const &) -> AtomicGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AtomicGenericSycl &&) -> AtomicGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AtomicGenericSycl() = default;
    };

    namespace detail
    {
        template <typename THierarchy>
        struct SyclMemoryScope {};

        template <>
        struct SyclMemoryScope<hierarchy::Grids>
        {
            static constexpr auto value = cl::sycl::ONEAPI::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Blocks>
        {
            static constexpr auto value = cl::sycl::ONEAPI::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Threads>
        {
            static constexpr auto value = cl::sycl::ONEAPI::memory_scope::work_group;
        };

        template <typename THierarchy>
        struct SyclMemorySpace {};

        template <>
        struct SyclMemorySpace<hierarchy::Grids>
        {
            static constexpr auto value = cl::sycl::access::address_space::global_device_space;
        };

        template <>
        struct SyclMemorySpace<hierarchy::Blocks>
        {
            static constexpr auto value = cl::sycl::access::address_space::global_device_space;
        };

        template <>
        struct SyclMemorySpace<hierarchy::Threads>
        {
            static constexpr auto value = cl::sycl::access::address_space::local_space;
        };

        template <typename T, typename THierarchy>
        using atomic_ref = cl::sycl::ONEAPI::atomic_ref<T, cl::sycl::ONEAPI::memory_order_relaxed,
                                                        SyclMemoryScope<THierarchy>::value,
                                                        SyclMemorySpace<THierarchy>::value>;
    }

    namespace traits
    {
        //-----------------------------------------------------------------------------
        // Add.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");

            //-----------------------------------------------------------------------------
            //
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_add(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Sub.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_sub(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Min.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_min(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Max.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_max(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Exch.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.exchange(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Inc.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                using Integral = std::conditional_t<std::is_integral_v<T>, T,
                                                    std::conditional_t<std::is_same_v<double, T>, unsigned long long,
                                                    std::conditional_t<std::is_same_v<float, T>, unsigned int,
                                                    std::conditional_t<std::is_same_v<cl::sycl::half, T>, unsigned short, void>>>>;
                
                auto ref = alpaka::detail::atomic_ref<Integral, THierarchy>{reinterpret_cast<Integral*>(*addr)};

                auto old = ref.load();
                auto assumed = Integral{};

                do
                {
                    assumed = old;
                    const auto old_val = *(reinterpret_cast<T*>(&assumed));
                    const auto new_val = (old_val >= value) ? 0 : (old_val + 1);
                    old = ref.compare_exchange_strong(assumed, new_val);
                } while(assumed != old);


                return *(reinterpret_cast<T*>(&old));
            }
        };

        //-----------------------------------------------------------------------------
        // Dec.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                using Integral = std::conditional_t<std::is_integral_v<T>, T,
                                                    std::conditional_t<std::is_same_v<double, T>, unsigned long long,
                                                    std::conditional_t<std::is_same_v<float, T>, unsigned int,
                                                    std::conditional_t<std::is_same_v<cl::sycl::half, T>, unsigned short, void>>>>;
                
                auto ref = alpaka::detail::atomic_ref<Integral, THierarchy>{reinterpret_cast<Integral*>(*addr)};

                auto old = ref.load();
                auto assumed = Integral{};

                do
                {
                    assumed = old;
                    const auto old_val = *(reinterpret_cast<T*>(&assumed));
                    const auto new_val = ((old_val == 0) || (old_val > value)) ? value : (old_val - 1);
                    old = ref.compare_exchange_strong(assumed, new_val);
                } while(assumed != old);


                return *(reinterpret_cast<T*>(&old));
            }
        };

        //-----------------------------------------------------------------------------
        // And.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T>, "Bitwise operations are not supported on non-integral types.");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_and(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Or.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T>, "Bitwise operations are not supported on non-integral types.");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_or(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Xor.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T>, "Bitwise operations are not supported on non-integral types.");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};
                return ref.fetch_xor(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Cas.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicGenericSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicGenericSycl const &, T * const addr, T const & compare, T const & value) -> T
            {
                auto ref = alpaka::detail::atomic_ref<T, THierarchy>{*addr};

                // SYCL stores the value in *addr to the "compare" parameter if the values are not equal. Since
                // alpaka's interface does not expect this we need to copy "compare" to this function and forget it
                // afterwards.
                auto tmp = compare;

                // We always want to return the old value at the end.
                const auto old = ref.load();

                // This returns a bool telling us if the exchange happened or not. Useless in this case.
                ref.compare_exchange_strong(tmp, value);

                return old;
            }
        };
    }
}

#endif
