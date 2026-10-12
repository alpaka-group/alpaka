/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

namespace alpaka::simd
{
    template<typename T, typename TAcc>
    class PortableSimd;
} // namespace alpaka::simd

#include <array>
#include <cstdint>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

// Forward declarations for SYCL accelerator types
namespace alpaka
{
    struct TagCpuSycl;
    struct TagGpuSyclIntel;
    struct TagGpuSyclHipAmd;
    struct TagGpuSyclGeneric;

    template<typename TTag, typename TDim, typename TIdx>
    class AccGenericSycl;
} // namespace alpaka

namespace alpaka::simd
{
    //! SYCL SIMD implementation using scalar fallback with compiler auto-vectorization
    //! This provides a portable SIMD interface for SYCL accelerators
    template<typename T, typename TTag, typename TDim, typename TIdx>
    class PortableSimd<T, AccGenericSycl<TTag, TDim, TIdx>>
    {
    public:
        using value_type = T;
        using accelerator_type = AccGenericSycl<TTag, TDim, TIdx>;

        // Unified SIMD width via SimdPolicySelection - single source of truth
        static constexpr std::size_t simd_size
            = SimdPolicySelection<AccGenericSycl<TTag, TDim, TIdx>>::template width<T>();

    private:
        std::array<T, simd_size> m_data{};

    public:
        //! Default constructor - zero-initialize
        ALPAKA_FN_ACC PortableSimd() = default;

        //! Broadcast constructor
        ALPAKA_FN_ACC explicit PortableSimd(T value)
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                m_data[i] = value;
            }
        }

        //! Array constructor
        ALPAKA_FN_ACC explicit PortableSimd(std::array<T, simd_size> const& data) : m_data(data)
        {
        }

        //! Copy constructor
        ALPAKA_FN_ACC PortableSimd(PortableSimd const&) = default;

        //! Move constructor
        ALPAKA_FN_ACC PortableSimd(PortableSimd&&) = default;

        //! Copy assignment
        ALPAKA_FN_ACC PortableSimd& operator=(PortableSimd const&) = default;

        //! Move assignment
        ALPAKA_FN_ACC PortableSimd& operator=(PortableSimd&&) = default;

        //! Destructor
        ALPAKA_FN_ACC ~PortableSimd() = default;

        //! Get SIMD width
        ALPAKA_FN_ACC static constexpr std::size_t size()
        {
            return simd_size;
        }

        //! Element access
        ALPAKA_FN_ACC T& operator[](std::size_t i)
        {
            return m_data[i];
        }

        ALPAKA_FN_ACC T const& operator[](std::size_t i) const
        {
            return m_data[i];
        }

        //! Load from memory (aligned)
        ALPAKA_FN_ACC void load(T const* ptr)
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                m_data[i] = ptr[i];
            }
        }

        //! Store to memory (aligned)
        ALPAKA_FN_ACC void store(T* ptr) const
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                ptr[i] = m_data[i];
            }
        }

        //! Arithmetic operators
        ALPAKA_FN_ACC PortableSimd operator+(PortableSimd const& other) const
        {
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] + other.m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC PortableSimd operator-(PortableSimd const& other) const
        {
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] - other.m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC PortableSimd operator*(PortableSimd const& other) const
        {
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] * other.m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC PortableSimd operator/(PortableSimd const& other) const
        {
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] / other.m_data[i];
            }
            return result;
        }

        //! Bitwise operators (for integer types)
        ALPAKA_FN_ACC PortableSimd operator&(PortableSimd const& other) const
        {
            static_assert(std::is_integral_v<T>, "Bitwise AND requires integral types");
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] & other.m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC PortableSimd operator|(PortableSimd const& other) const
        {
            static_assert(std::is_integral_v<T>, "Bitwise OR requires integral types");
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] | other.m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC PortableSimd operator^(PortableSimd const& other) const
        {
            static_assert(std::is_integral_v<T>, "Bitwise XOR requires integral types");
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] ^ other.m_data[i];
            }
            return result;
        }

        //! Shift operators (for integer types)
        ALPAKA_FN_ACC PortableSimd operator<<(int shift) const
        {
            static_assert(std::is_integral_v<T>, "Left shift requires integral types");
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] << shift;
            }
            return result;
        }

        ALPAKA_FN_ACC PortableSimd operator>>(int shift) const
        {
            static_assert(std::is_integral_v<T>, "Right shift requires integral types");
            PortableSimd result;
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                result.m_data[i] = m_data[i] >> shift;
            }
            return result;
        }

        //! Compound assignment operators
        ALPAKA_FN_ACC PortableSimd& operator+=(PortableSimd const& other)
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                m_data[i] += other.m_data[i];
            }
            return *this;
        }

        ALPAKA_FN_ACC PortableSimd& operator-=(PortableSimd const& other)
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                m_data[i] -= other.m_data[i];
            }
            return *this;
        }

        ALPAKA_FN_ACC PortableSimd& operator*=(PortableSimd const& other)
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                m_data[i] *= other.m_data[i];
            }
            return *this;
        }

        ALPAKA_FN_ACC PortableSimd& operator/=(PortableSimd const& other)
        {
            for(std::size_t i = 0; i < simd_size; ++i)
            {
                m_data[i] /= other.m_data[i];
            }
            return *this;
        }

        //! Reduction operations
        ALPAKA_FN_ACC T sum() const
        {
            T result = m_data[0];
            for(std::size_t i = 1; i < simd_size; ++i)
            {
                result += m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC T min() const
        {
            T result = m_data[0];
            for(std::size_t i = 1; i < simd_size; ++i)
            {
                if(m_data[i] < result)
                    result = m_data[i];
            }
            return result;
        }

        ALPAKA_FN_ACC T max() const
        {
            T result = m_data[0];
            for(std::size_t i = 1; i < simd_size; ++i)
            {
                if(m_data[i] > result)
                    result = m_data[i];
            }
            return result;
        }
    };

} // namespace alpaka::simd

#endif // ALPAKA_ACC_SYCL_ENABLED
