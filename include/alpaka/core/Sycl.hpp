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

#include <alpaka/elem/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <CL/sycl.hpp>

#include <array>
#include <type_traits>
#include <utility>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstddef>

//-----------------------------------------------------------------------------
// SYCL vector types trait specializations.
namespace alpaka
{
    namespace detail
    {
        // Remove std::is_same boilerplate
        template <typename T, typename... Ts>
        struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

        // Extract cl::sycl::vec's number of elements which isn't constexpr
        // despite being a template parameter
        template <int N_arg>
        struct val { static constexpr auto N = N_arg; };

        template <typename T, int N>
        constexpr auto extract_impl(const cl::sycl::vec<T, N>&) -> val<N>;

        template <typename T>
        constexpr auto extract = decltype(extract_impl(std::declval<T>()))::N;
    }

    namespace traits
    {
        //##################################################################
        //! In contrast to CUDA SYCL doesn't know 1D vectors. It does
        //! support OpenCL's data types which have additional requirements
        //! on top of those in the C++ standard. Note that SYCL's equivalent
        //! to CUDA's dim3 type is a different class type and thus not used
        //! here.
        template<typename T>
        struct IsSyclBuiltInType :
            alpaka::detail::is_any<T,
                // built-in scalar types - these are the standard C++ built-in types, std::size_t, std::byte and cl::sycl::half
                cl::sycl::half,

                // 2 component vector types
                cl::sycl::char2, cl::sycl::schar2, cl::sycl::uchar2,
                cl::sycl::short2, cl::sycl::ushort2,
                cl::sycl::int2, cl::sycl::uint2,
                cl::sycl::long2, cl::sycl::ulong2,
                cl::sycl::longlong2, cl::sycl::ulonglong2,
                cl::sycl::float2, cl::sycl::double2, cl::sycl::half2,
                cl::sycl::cl_char2, cl::sycl::cl_uchar2,
                cl::sycl::cl_short2, cl::sycl::cl_ushort2,
                cl::sycl::cl_int2, cl::sycl::cl_uint2,
                cl::sycl::cl_long2, cl::sycl::cl_ulong2,
                cl::sycl::cl_float2, cl::sycl::cl_double2, cl::sycl::cl_half2,

                // 3 component vector types
                cl::sycl::char3, cl::sycl::schar3, cl::sycl::uchar3,
                cl::sycl::short3, cl::sycl::ushort3,
                cl::sycl::int3, cl::sycl::uint3,
                cl::sycl::long3, cl::sycl::ulong3,
                cl::sycl::longlong3, cl::sycl::ulonglong3,
                cl::sycl::float3, cl::sycl::double3, cl::sycl::half3,
                cl::sycl::cl_char3, cl::sycl::cl_uchar3,
                cl::sycl::cl_short3, cl::sycl::cl_ushort3,
                cl::sycl::cl_int3, cl::sycl::cl_uint3,
                cl::sycl::cl_long3, cl::sycl::cl_ulong3,
                cl::sycl::cl_float3, cl::sycl::cl_double3, cl::sycl::cl_half3,

                // 4 component vector types
                cl::sycl::char4, cl::sycl::schar4, cl::sycl::uchar4,
                cl::sycl::short4, cl::sycl::ushort4,
                cl::sycl::int4, cl::sycl::uint4,
                cl::sycl::long4, cl::sycl::ulong4,
                cl::sycl::longlong4, cl::sycl::ulonglong4,
                cl::sycl::float4, cl::sycl::double4, cl::sycl::half4,
                cl::sycl::cl_char4, cl::sycl::cl_uchar4,
                cl::sycl::cl_short4, cl::sycl::cl_ushort4,
                cl::sycl::cl_int4, cl::sycl::cl_uint4,
                cl::sycl::cl_long4, cl::sycl::cl_ulong4,
                cl::sycl::cl_float4, cl::sycl::cl_double4, cl::sycl::cl_half4,

                // 8 component vector types
                cl::sycl::char8, cl::sycl::schar8, cl::sycl::uchar8,
                cl::sycl::short8, cl::sycl::ushort8,
                cl::sycl::int8, cl::sycl::uint8,
                cl::sycl::long8, cl::sycl::ulong8,
                cl::sycl::longlong8, cl::sycl::ulonglong8,
                cl::sycl::float8, cl::sycl::double8, cl::sycl::half8,
                cl::sycl::cl_char8, cl::sycl::cl_uchar8,
                cl::sycl::cl_short8, cl::sycl::cl_ushort8,
                cl::sycl::cl_int8, cl::sycl::cl_uint8,
                cl::sycl::cl_long8, cl::sycl::cl_ulong8,
                cl::sycl::cl_float8, cl::sycl::cl_double8, cl::sycl::cl_half8,

                // 16 component vector types
                cl::sycl::char16, cl::sycl::schar16, cl::sycl::uchar16,
                cl::sycl::short16, cl::sycl::ushort16,
                cl::sycl::int16, cl::sycl::uint16,
                cl::sycl::long16, cl::sycl::ulong16,
                cl::sycl::longlong16, cl::sycl::ulonglong16,
                cl::sycl::float16, cl::sycl::double16, cl::sycl::half16,
                cl::sycl::cl_char16, cl::sycl::cl_uchar16,
                cl::sycl::cl_short16, cl::sycl::cl_ushort16,
                cl::sycl::cl_int16, cl::sycl::cl_uint16,
                cl::sycl::cl_long16, cl::sycl::cl_ulong16,
                cl::sycl::cl_float16, cl::sycl::cl_double16, cl::sycl::cl_half16
            >
        {};

        //##################################################################
        //! SYCL's types get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<IsSyclBuiltInType<T>::value>>
        {
            using type = DimInt<alpaka::detail::extract<T>>;
        };

        //##################################################################
        //! The SYCL vectors' elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<IsSyclBuiltInType<T>::value>>
        {
            using type = std::conditional_t<std::is_scalar_v<T>, T, typename T::element_type>;
        };
    }

    namespace extent
    {
        namespace traits
        {
            //##################################################################
            //! The SYCL vectors' extent get trait specialization.
            template<typename TExtent>
            struct GetExtent<DimInt<Dim<TExtent>::value>, TExtent,
                             std::enable_if_t<alpaka::traits::IsSyclBuiltInType<TExtent>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                {
                    if constexpr(std::is_scalar_v<TExtent>)
                        return extent;
                    else
                        return extent.template swizzle<DimInt<Dim<TExtent>::value>::value>();
                }
            };

            //#############################################################################
            //! The SYCL vectors' extent set trait specialization.
            template<typename TExtent, typename TExtentVal>
            struct SetExtent<DimInt<Dim<TExtent>::value>, TExtent, TExtentVal,
                             std::enable_if_t<alpaka::traits::IsSyclBuiltInType<TExtent>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                {
                    if constexpr(std::is_scalar_v<TExtent>)
                        extent = extentVal;
                    else
                        extent.template swizzle<DimInt<Dim<TExtent>::value>::value>() = extentVal;
                }
            };
        }
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL vectors' offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<DimInt<Dim<TOffsets>::value>, TOffsets,
                         std::enable_if_t<IsSyclBuiltInType<TOffsets>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const & offsets)
            {
                if constexpr(std::is_scalar_v<TOffsets>)
                    return offsets;
                else
                    return offsets.template swizzle<DimInt<Dim<TOffsets>::value>::value>();
            }
        };

        //#############################################################################
        //! The SYCL vectors' offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<DimInt<Dim<TOffsets>::value>, TOffsets, TOffset,
                         std::enable_if_t<IsSyclBuiltInType<TOffsets>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const & offsets, TOffset const & offset)
            {
                if constexpr(std::is_scalar_v<TOffsets>)
                    offsets = offset;
                else
                    offsets.template swizzle<DimInt<Dim<TOffsets>::value>::value>() = offset;
            }
        };

        //#############################################################################
        //! The SYCL vectors' idx type trait specialization.
        template<typename TIdx>
        struct IdxType<TIdx, std::enable_if_t<IsSyclBuiltInType<TIdx>::value>>
        {
            using type = std::size_t;
        };
    }
}

#endif
