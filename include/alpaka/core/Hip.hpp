/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/BoostPredef.hpp>

#    if !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/elem/Traits.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/offset/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <hip/hip_runtime.h>

#    include <cstddef>
#    include <type_traits>
#    include <utility>

#    if BOOST_COMP_HIP
#        define HIPRT_CB
#    endif

#    define ALPAKA_PP_CONCAT_DO(X, Y) X##Y
#    define ALPAKA_PP_CONCAT(X, Y) ALPAKA_PP_CONCAT_DO(X, Y)
//! prefix a name with `hip`
#    define ALPAKA_API_PREFIX(name) ALPAKA_PP_CONCAT_DO(hip, name)

// HIP vector_types.h trait specializations.
namespace alpaka
{
    //! The HIP specifics.
    namespace hip::traits
    {
        //! The HIP vectors 1D dimension get trait specialization.
        template<typename T>
        struct IsHipBuiltInType
            : std::integral_constant<bool, std::is_same_v<T, char1> || std::is_same_v<T, double1> || std::is_same_v<T, float1> || std::is_same_v<T, int1> || std::is_same_v<T, long1> || std::is_same_v<T, longlong1> || std::is_same_v<T, short1> || std::is_same_v<T, uchar1> || std::is_same_v<T, uint1> || std::is_same_v<T, ulong1> || std::is_same_v<T, ulonglong1> || std::is_same_v<T, ushort1> || std::is_same_v<T, char2> || std::is_same_v<T, double2> || std::is_same_v<T, float2> || std::is_same_v<T, int2> || std::is_same_v<T, long2> || std::is_same_v<T, longlong2> || std::is_same_v<T, short2> || std::is_same_v<T, uchar2> || std::is_same_v<T, uint2> || std::is_same_v<T, ulong2> || std::is_same_v<T, ulonglong2> || std::is_same_v<T, ushort2> || std::is_same_v<T, char3> || std::is_same_v<T, dim3> || std::is_same_v<T, double3> || std::is_same_v<T, float3> || std::is_same_v<T, int3> || std::is_same_v<T, long3> || std::is_same_v<T, longlong3> || std::is_same_v<T, short3> || std::is_same_v<T, uchar3> || std::is_same_v<T, uint3> || std::is_same_v<T, ulong3> || std::is_same_v<T, ulonglong3> || std::is_same_v<T, ushort3> || std::is_same_v<T, char4> || std::is_same_v<T, double4> || std::is_same_v<T, float4> || std::is_same_v<T, int4> || std::is_same_v<T, long4> || std::is_same_v<T, longlong4> || std::is_same_v<T, short4> || std::is_same_v<T, uchar4> || std::is_same_v<T, uint4> || std::is_same_v<T, ulong4> || std::is_same_v<T, ulonglong4> || std::is_same_v<T, ushort4>>
        {
        };
    } // namespace hip::traits
    namespace traits
    {
        // If you receive '"alpaka::traits::DimType" has already been defined'
        // then too many operators in the enable_if are used. Split them in two or more structs.
        // (compiler: gcc 5.3.0)
        //! The HIP vectors 1D dimension get trait specialization.
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    char1> || std::is_same_v<T, double1> || std::is_same_v<T, float1> || std::is_same_v<T, int1> || std::is_same_v<T, long1> || std::is_same_v<T, longlong1> || std::is_same_v<T, short1>>>
        {
            using type = DimInt<1u>;
        };
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    uchar1> || std::is_same_v<T, uint1> || std::is_same_v<T, ulong1> || std::is_same_v<T, ulonglong1> || std::is_same_v<T, ushort1>>>
        {
            using type = DimInt<1u>;
        };
        //! The HIP vectors 2D dimension get trait specialization.
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    char2> || std::is_same_v<T, double2> || std::is_same_v<T, float2> || std::is_same_v<T, int2> || std::is_same_v<T, long2> || std::is_same_v<T, longlong2> || std::is_same_v<T, short2>>>
        {
            using type = DimInt<2u>;
        };
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    uchar2> || std::is_same_v<T, uint2> || std::is_same_v<T, ulong2> || std::is_same_v<T, ulonglong2> || std::is_same_v<T, ushort2>>>
        {
            using type = DimInt<2u>;
        };
        //! The HIP vectors 3D dimension get trait specialization.
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    char3> || std::is_same_v<T, dim3> || std::is_same_v<T, double3> || std::is_same_v<T, float3> || std::is_same_v<T, int3> || std::is_same_v<T, long3> || std::is_same_v<T, longlong3> || std::is_same_v<T, short3>>>
        {
            using type = DimInt<3u>;
        };
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    uchar3> || std::is_same_v<T, uint3> || std::is_same_v<T, ulong3> || std::is_same_v<T, ulonglong3> || std::is_same_v<T, ushort3>>>
        {
            using type = DimInt<3u>;
        };
        //! The HIP vectors 4D dimension get trait specialization.
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    char4> || std::is_same_v<T, double4> || std::is_same_v<T, float4> || std::is_same_v<T, int4> || std::is_same_v<T, long4> || std::is_same_v<T, longlong4> || std::is_same_v<T, short4>>>
        {
            using type = DimInt<4u>;
        };
        template<typename T>
        struct DimType<
            T,
            std::enable_if_t<
                std::is_same_v<
                    T,
                    uchar4> || std::is_same_v<T, uint4> || std::is_same_v<T, ulong4> || std::is_same_v<T, ulonglong4> || std::is_same_v<T, ushort4>>>
        {
            using type = DimInt<4u>;
        };

        //! The HIP vectors elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<hip::traits::IsHipBuiltInType<T>::value>>
        {
            using type = decltype(std::declval<T>().x);
        };

        //! The HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 1u>,
            TExtent,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.x;
            }
        };
        //! The HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 2u>,
            TExtent,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.y;
            }
        };
        //! The HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 3u>,
            TExtent,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.z;
            }
        };
        //! The HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 4u>,
            TExtent,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.w;
            }
        };
        //! The HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 1u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.x = extentVal;
            }
        };
        //! The HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 2u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.y = extentVal;
            }
        };
        //! The HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 3u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.z = extentVal;
            }
        };
        //! The HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 4u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TExtent>::value && (Dim<TExtent>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.w = extentVal;
            }
        };

        //! The HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 1u>,
            TOffsets,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.x;
            }
        };
        //! The HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 2u>,
            TOffsets,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.y;
            }
        };
        //! The HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 3u>,
            TOffsets,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.z;
            }
        };
        //! The HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 4u>,
            TOffsets,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.w;
            }
        };
        //! The HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 1u>,
            TOffsets,
            TOffset,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.x = offset;
            }
        };
        //! The HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 2u>,
            TOffsets,
            TOffset,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.y = offset;
            }
        };
        //! The HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 3u>,
            TOffsets,
            TOffset,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.z = offset;
            }
        };
        //! The HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 4u>,
            TOffsets,
            TOffset,
            std::enable_if_t<hip::traits::IsHipBuiltInType<TOffsets>::value && (Dim<TOffsets>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.w = offset;
            }
        };

        //! The HIP vectors idx type trait specialization.
        template<typename TIdx>
        struct IdxType<TIdx, std::enable_if_t<hip::traits::IsHipBuiltInType<TIdx>::value>>
        {
            using type = std::size_t;
        };
    } // namespace traits
} // namespace alpaka

#    include <alpaka/core/UniformCudaHip.hpp>

#endif
