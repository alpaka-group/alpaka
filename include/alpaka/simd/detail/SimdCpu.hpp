/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Alpaka Simd CPU Implementation

#include "alpaka/acc/AccCpuSerial.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Assert.hpp"

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
#    include "alpaka/acc/AccCpuThreads.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    include "alpaka/acc/AccCpuOmp2Blocks.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
#    include "alpaka/acc/AccCpuOmp2Threads.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#    include "alpaka/acc/AccCpuTbbBlocks.hpp"
#endif

// Explicit SIMD policy selection & environment detection
#include "alpaka/simd/SimdPolicySelection.hpp"

#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>

// Provide only the namespace alias when std::experimental::simd is technically available and set ON at cmake(default)
#if ALPAKA_USE_STD_SIMD && ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE
namespace stdx = std::experimental;
#endif

// Forward declaration
namespace alpaka::simd
{
    template<typename T, typename TAcc>
    class PortableSimd;

    namespace detail
    {

        // Concept: Valid SIMD element type
        template<typename T>
        concept SimdElementType = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

        // Concept: Integral types for bitwise operations
        template<typename T>
        concept IntegralSimdType = SimdElementType<T> && std::is_integral_v<T>;

    } // namespace detail

    // ---------------------------------------------------------------------
    // Restrict macro (host+device friendly) for pointer arguments to help
    // the compiler assume non-aliasing in load/store of SIMD containers.
    // ---------------------------------------------------------------------
#ifndef ALPAKA_SIMD_RESTRICT
#    if defined(_MSC_VER)
#        define ALPAKA_SIMD_RESTRICT __restrict
#    else
#        define ALPAKA_SIMD_RESTRICT __restrict__
#    endif
#endif

#if ALPAKA_USE_STD_SIMD && ALPAKA_SIMD_ENV_STD_EXP_SIMD_AVAILABLE
#    if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    // Template specialization for std::experimental::simd (only when serial CPU backend enabled)
    template<typename T, typename TDim, typename TIdx>
    requires detail::SimdElementType<T>
    class PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        stdx::native_simd<T> data;

        ALPAKA_FN_ACC constexpr explicit PortableSimd(stdx::native_simd<T> const& simd_data) : data(simd_data)
        {
        }

    public:
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr std::size_t size() noexcept
        {
            return stdx::native_simd<T>::size();
        }

        ALPAKA_FN_ACC constexpr PortableSimd() : data(stdx::native_simd<T>(T{0}))
        {
        }

        ALPAKA_FN_ACC constexpr explicit PortableSimd(T scalar) : data(scalar)
        {
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void load(T const* ALPAKA_SIMD_RESTRICT ptr) noexcept
        {
            data.copy_from(ptr, stdx::element_aligned);
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void store(T* ALPAKA_SIMD_RESTRICT ptr) const noexcept
        {
            data.copy_to(ptr, stdx::element_aligned);
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator+(PortableSimd const& other) const noexcept
        {
            return PortableSimd(data + other.data);
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator-(PortableSimd const& other) const noexcept
        {
            return PortableSimd(data - other.data);
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator*(PortableSimd const& other) const noexcept
        {
            return PortableSimd(data * other.data);
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator/(PortableSimd const& other) const noexcept
        {
            return PortableSimd(data / other.data);
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd& operator+=(PortableSimd const& other) noexcept
        {
            data += other.data;
            return *this;
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd& operator-=(PortableSimd const& other) noexcept
        {
            data -= other.data;
            return *this;
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd& operator*=(PortableSimd const& other) noexcept
        {
            data *= other.data;
            return *this;
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd& operator/=(PortableSimd const& other) noexcept
        {
            data /= other.data;
            return *this;
        }

        // Concept-constrained bitwise operations
        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator>>(PortableSimd const& shift) const noexcept
        {
            return PortableSimd(data >> shift.data);
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator>>(T shift) const noexcept
        {
            return PortableSimd(data >> shift);
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator<<(PortableSimd const& shift) const noexcept
        {
            return PortableSimd(data << shift.data);
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator<<(T shift) const noexcept
        {
            return PortableSimd(data << shift);
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator&(PortableSimd const& mask) const noexcept
        {
            return PortableSimd(data & mask.data);
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator|(PortableSimd const& mask) const noexcept
        {
            return PortableSimd(data | mask.data);
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd operator^(PortableSimd const& mask) const noexcept
        {
            return PortableSimd(data ^ mask.data);
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE T sum() const noexcept
        {
            return stdx::reduce(data);
        }

        // Lane access (read-only). Mutable access not required after tail helpers switched to buffer path.
        ALPAKA_FN_ACC ALPAKA_FN_INLINE T operator[](std::size_t idx) const noexcept
        {
            ALPAKA_ASSERT_ACC(idx < data.size());
            return data[idx];
        }

        template<typename F>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd map(F&& f) const
        {
            PortableSimd r;
            for(std::size_t i = 0; i < data.size(); ++i)
            {
                r.data[i] = f(data[i]);
            }
            return r;
        }
    };

#    endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#else // fallback PortableSimd implementation
#    if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    // Lean fixed-width array fallback tuned
    template<typename T, typename TDim, typename TIdx>
    requires detail::SimdElementType<T>
    class PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        // Unified width directly from SimdPolicySelection. Single source of truth.
        static constexpr std::size_t WIDTH = SimdPolicySelection<AccCpuSerial<TDim, TIdx>>::template width<T>();

        T data[WIDTH];

        // Loop hint macros (host-side only). Avoid pragmas that trigger warnings under NVCC/Clang.
        // - NVCC: suppress GCC-specific pragmas (#pragma GCC unroll) -> use plain 'unroll' or nothing.
        // - Clang: safe to request vectorize+unroll.
        // Keep conservative to guarantee warning-free CI.
#        if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
#            if defined(__CUDACC__) || defined(__NVCC__)
        // NVCC host compilation: array-based fallback with '#pragma unroll 8'
#                define ALPAKA_SIMD_FALLBACK_LOOP_HINT _Pragma("unroll 8")
#            elif ALPAKA_COMP_CLANG
#                define ALPAKA_SIMD_FALLBACK_LOOP_HINT                                                                \
                    _Pragma("clang loop vectorize(enable)") _Pragma("clang loop unroll_count(8)")
#            elif ALPAKA_COMP_GNUC
        /* GCC: Vectorization hints for array-based fallback performance */
#                define ALPAKA_SIMD_FALLBACK_LOOP_HINT _Pragma("GCC ivdep") _Pragma("GCC unroll 8")
#            else
#                define ALPAKA_SIMD_FALLBACK_LOOP_HINT
#            endif
#        else
#            define ALPAKA_SIMD_FALLBACK_LOOP_HINT
#        endif
        // Unified FOR macro applying the loop hint (WIDTH guaranteed small).
#        define ALPAKA_SIMD_FALLBACK_FOR ALPAKA_SIMD_FALLBACK_LOOP_HINT for(std::size_t i = 0; i < WIDTH; ++i)

    public:
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr std::size_t size() noexcept
        {
            return WIDTH;
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr PortableSimd() noexcept : data{}
        {
            for(std::size_t i = 0; i < WIDTH; ++i)
                data[i] = T{0};
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr explicit PortableSimd(T v) noexcept : data{}
        {
            for(std::size_t i = 0; i < WIDTH; ++i)
                data[i] = v;
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE void load(T const* ALPAKA_SIMD_RESTRICT ptr) noexcept
        {
            ALPAKA_SIMD_FALLBACK_FOR
            {
                data[i] = ptr[i];
            }
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE void store(T* ALPAKA_SIMD_RESTRICT ptr) const noexcept
        {
            ALPAKA_SIMD_FALLBACK_FOR
            {
                ptr[i] = data[i];
            }
        }

        // Lane accessors (const & mutable) used by tail-hiding helpers.
        ALPAKA_FN_ACC ALPAKA_FN_INLINE T operator[](std::size_t idx) const noexcept
        {
            ALPAKA_ASSERT_ACC(idx < WIDTH);
            return data[idx];
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE T& operator[](std::size_t idx) noexcept
        {
            ALPAKA_ASSERT_ACC(idx < WIDTH);
            return data[idx];
        }

#        define ALPAKA_SIMD_FALLBACK_BINOP(OP)                                                                        \
            ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd operator OP(PortableSimd const& o) const noexcept             \
            {                                                                                                         \
                PortableSimd r;                                                                                       \
                ALPAKA_SIMD_FALLBACK_FOR                                                                              \
                {                                                                                                     \
                    r.data[i] = data[i] OP o.data[i];                                                                 \
                }                                                                                                     \
                return r;                                                                                             \
            }                                                                                                         \
            ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd& operator OP##=(PortableSimd const& o) noexcept               \
            {                                                                                                         \
                ALPAKA_SIMD_FALLBACK_FOR                                                                              \
                {                                                                                                     \
                    data[i] OP## = o.data[i];                                                                         \
                }                                                                                                     \
                return *this;                                                                                         \
            }
        ALPAKA_SIMD_FALLBACK_BINOP(+)
        ALPAKA_SIMD_FALLBACK_BINOP(-)
        ALPAKA_SIMD_FALLBACK_BINOP(*)
        ALPAKA_SIMD_FALLBACK_BINOP(/)
        ALPAKA_SIMD_FALLBACK_BINOP(&)
        ALPAKA_SIMD_FALLBACK_BINOP(|)
        ALPAKA_SIMD_FALLBACK_BINOP(^)
#        undef ALPAKA_SIMD_FALLBACK_BINOP

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd operator>>(PortableSimd const& o) const noexcept
        {
            PortableSimd r;
            ALPAKA_SIMD_FALLBACK_FOR
            {
                r.data[i] = data[i] >> o.data[i];
            }
            return r;
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd operator>>(T s) const noexcept
        {
            PortableSimd r;
            ALPAKA_SIMD_FALLBACK_FOR
            {
                r.data[i] = data[i] >> s;
            }
            return r;
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd operator<<(PortableSimd const& o) const noexcept
        {
            PortableSimd r;
            ALPAKA_SIMD_FALLBACK_FOR
            {
                r.data[i] = data[i] << o.data[i];
            }
            return r;
        }

        template<detail::IntegralSimdType U = T>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd operator<<(T s) const noexcept
        {
            PortableSimd r;
            ALPAKA_SIMD_FALLBACK_FOR
            {
                r.data[i] = data[i] << s;
            }
            return r;
        }

        ALPAKA_FN_ACC ALPAKA_FN_INLINE T sum() const noexcept
        {
            T r = T{0};
            ALPAKA_SIMD_FALLBACK_FOR
            {
                r += data[i];
            }
            return r;
        }

        template<typename F>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE PortableSimd map(F&& f) const
        {
            PortableSimd r;
            ALPAKA_SIMD_FALLBACK_FOR
            {
                r.data[i] = f(data[i]);
            }
            return r;
        }

#        undef ALPAKA_SIMD_FALLBACK_LOOP_HINT
#        undef ALPAKA_SIMD_FALLBACK_FOR
    };
#    endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#endif

} // namespace alpaka::simd
