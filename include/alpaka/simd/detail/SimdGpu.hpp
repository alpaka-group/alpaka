/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    include "alpaka/acc/AccGpuCudaRt.hpp"
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#    include "alpaka/acc/AccGpuHipRt.hpp"
#endif

#include "alpaka/simd/SimdPolicySelection.hpp"

#include <cassert>

namespace alpaka::simd
{
    template<typename T, typename TAcc>
    class PortableSimd;
} // namespace alpaka::simd

namespace alpaka::simd
{
    //! Define GPU SIMD implementation macro to avoid code duplication
    //! GPU SIMD uses scalar operations (SIMD width = 1) - parallelism through many threads
#define ALPAKA_DEFINE_GPU_SIMD_SPECIALIZATION(AccType)                                                                \
    template<typename T, typename TDim, typename TIdx>                                                                \
    class PortableSimd<T, AccType<TDim, TIdx>>                                                                        \
    {                                                                                                                 \
        T data;                                                                                                       \
                                                                                                                      \
    public:                                                                                                           \
        /*! SIMD width from unified SimdPolicySelection source of truth */                                            \
        constexpr std::size_t size() const                                                                            \
        {                                                                                                             \
            return SimdPolicySelection<AccType<TDim, TIdx>>::template width<T>();                                     \
        }                                                                                                             \
                                                                                                                      \
        /*! Default constructor - no initialization for performance */                                                \
        PortableSimd() = default;                                                                                     \
                                                                                                                      \
        /*! Zero constructor */                                                                                       \
        ALPAKA_FN_ACC explicit PortableSimd(std::nullptr_t) : data(T{0})                                              \
        {                                                                                                             \
        }                                                                                                             \
                                                                                                                      \
        /*! Scalar constructor */                                                                                     \
        ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar)                                                  \
        {                                                                                                             \
        }                                                                                                             \
                                                                                                                      \
        /*! Load single element */                                                                                    \
        ALPAKA_FN_ACC void load(T const* ptr)                                                                         \
        {                                                                                                             \
            data = *ptr;                                                                                              \
        }                                                                                                             \
                                                                                                                      \
        /*! Store single element */                                                                                   \
        ALPAKA_FN_ACC void store(T* ptr) const                                                                        \
        {                                                                                                             \
            *ptr = data;                                                                                              \
        }                                                                                                             \
                                                                                                                      \
        /*! Arithmetic operators (scalar operations) */                                                               \
        ALPAKA_FN_ACC PortableSimd operator+(PortableSimd const& other) const                                         \
        {                                                                                                             \
            return PortableSimd(data + other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_FN_ACC PortableSimd operator-(PortableSimd const& other) const                                         \
        {                                                                                                             \
            return PortableSimd(data - other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_FN_ACC PortableSimd operator*(PortableSimd const& other) const                                         \
        {                                                                                                             \
            return PortableSimd(data * other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_FN_ACC PortableSimd operator/(PortableSimd const& other) const                                         \
        {                                                                                                             \
            return PortableSimd(data / other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        /*! Compound assignment operators */                                                                          \
        ALPAKA_FN_ACC PortableSimd& operator+=(PortableSimd const& other)                                             \
        {                                                                                                             \
            data += other.data;                                                                                       \
            return *this;                                                                                             \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_FN_ACC PortableSimd& operator-=(PortableSimd const& other)                                             \
        {                                                                                                             \
            data -= other.data;                                                                                       \
            return *this;                                                                                             \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_FN_ACC PortableSimd& operator*=(PortableSimd const& other)                                             \
        {                                                                                                             \
            data *= other.data;                                                                                       \
            return *this;                                                                                             \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_FN_ACC PortableSimd& operator/=(PortableSimd const& other)                                             \
        {                                                                                                             \
            data /= other.data;                                                                                       \
            return *this;                                                                                             \
        }                                                                                                             \
                                                                                                                      \
        /*! Bitwise operators for integral types */                                                                   \
        template<typename U = T>                                                                                      \
        ALPAKA_FN_ACC std::enable_if_t<std::is_integral_v<U>, PortableSimd> operator>>(int shift) const               \
        {                                                                                                             \
            return PortableSimd(data >> shift);                                                                       \
        }                                                                                                             \
                                                                                                                      \
        template<typename U = T>                                                                                      \
        ALPAKA_FN_ACC std::enable_if_t<std::is_integral_v<U>, PortableSimd> operator<<(int shift) const               \
        {                                                                                                             \
            return PortableSimd(data << shift);                                                                       \
        }                                                                                                             \
                                                                                                                      \
        template<typename U = T>                                                                                      \
        ALPAKA_FN_ACC std::enable_if_t<std::is_integral_v<U>, PortableSimd> operator&(                                \
            PortableSimd const& other) const                                                                          \
        {                                                                                                             \
            return PortableSimd(data & other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        template<typename U = T>                                                                                      \
        ALPAKA_FN_ACC std::enable_if_t<std::is_integral_v<U>, PortableSimd> operator|(                                \
            PortableSimd const& other) const                                                                          \
        {                                                                                                             \
            return PortableSimd(data | other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        template<typename U = T>                                                                                      \
        ALPAKA_FN_ACC std::enable_if_t<std::is_integral_v<U>, PortableSimd> operator^(                                \
            PortableSimd const& other) const                                                                          \
        {                                                                                                             \
            return PortableSimd(data ^ other.data);                                                                   \
        }                                                                                                             \
                                                                                                                      \
        /*! Sum returns the single scalar value */                                                                    \
        ALPAKA_FN_ACC T sum() const                                                                                   \
        {                                                                                                             \
            return data;                                                                                              \
        }                                                                                                             \
                                                                                                                      \
        /*! Element access (only element 0 exists) */                                                                 \
        ALPAKA_FN_ACC T operator[](std::size_t) const                                                                 \
        {                                                                                                             \
            return data; /* idx should always be 0 */                                                                 \
        }                                                                                                             \
    };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    //! CUDA GPU SIMD specialization
    ALPAKA_DEFINE_GPU_SIMD_SPECIALIZATION(AccGpuCudaRt)
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    //! HIP GPU SIMD specialization
    ALPAKA_DEFINE_GPU_SIMD_SPECIALIZATION(AccGpuHipRt)
#endif

#undef ALPAKA_DEFINE_GPU_SIMD_SPECIALIZATION

} // namespace alpaka::simd
