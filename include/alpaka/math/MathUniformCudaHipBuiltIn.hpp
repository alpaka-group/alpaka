/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Decay.hpp>
#include <alpaka/core/UniformCudaHip.hpp>
#include <alpaka/core/Unreachable.hpp>
#include <alpaka/math/Traits.hpp>

#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::math
{
    //! The CUDA built in abs.
    class AbsUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAbs, AbsUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acos.
    class AcosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcos, AcosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asin.
    class AsinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsin, AsinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan.
    class AtanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan, AtanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan2.
    class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cbrt.
    class CbrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCbrt, CbrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in ceil.
    class CeilUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCeil, CeilUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cos.
    class CosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCos, CosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in erf.
    class ErfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathErf, ErfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in exp.
    class ExpUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathExp, ExpUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in floor.
    class FloorUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFloor, FloorUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fmod.
    class FmodUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFmod, FmodUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isfinite.
    class IsfiniteUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathIsfinite, IsfiniteUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isinf.
    class IsinfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsinf, IsinfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isnan.
    class IsnanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsnan, IsnanUniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log.
    class LogUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog, LogUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in max.
    class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in min.
    class MinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in pow.
    class PowUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathPow, PowUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in remainder.
    class RemainderUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA round.
    class RoundUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRound, RoundUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA rsqrt.
    class RsqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sin.
    class SinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sincos.
    class SinCosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sqrt.
    class SqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tan.
    class TanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTan, TanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA trunc.
    class TruncUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTrunc, TruncUniformCudaHipBuiltIn>
    {
    };

    //! The standard library math trait specializations.
    class MathUniformCudaHipBuiltIn
        : public AbsUniformCudaHipBuiltIn
        , public AcosUniformCudaHipBuiltIn
        , public AsinUniformCudaHipBuiltIn
        , public AtanUniformCudaHipBuiltIn
        , public Atan2UniformCudaHipBuiltIn
        , public CbrtUniformCudaHipBuiltIn
        , public CeilUniformCudaHipBuiltIn
        , public CosUniformCudaHipBuiltIn
        , public ErfUniformCudaHipBuiltIn
        , public ExpUniformCudaHipBuiltIn
        , public FloorUniformCudaHipBuiltIn
        , public FmodUniformCudaHipBuiltIn
        , public LogUniformCudaHipBuiltIn
        , public MaxUniformCudaHipBuiltIn
        , public MinUniformCudaHipBuiltIn
        , public PowUniformCudaHipBuiltIn
        , public RemainderUniformCudaHipBuiltIn
        , public RoundUniformCudaHipBuiltIn
        , public RsqrtUniformCudaHipBuiltIn
        , public SinUniformCudaHipBuiltIn
        , public SinCosUniformCudaHipBuiltIn
        , public SqrtUniformCudaHipBuiltIn
        , public TanUniformCudaHipBuiltIn
        , public TruncUniformCudaHipBuiltIn
        , public IsnanUniformCudaHipBuiltIn
        , public IsinfUniformCudaHipBuiltIn
        , public IsfiniteUniformCudaHipBuiltIn
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
#            include <cuda_runtime.h>
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
#            include <hip/math_functions.h>
#        endif

    namespace trait
    {
        //! The CUDA built in abs trait specialization.
        template<typename TArg>
        struct Abs<AbsUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_signed_v<TArg>>>
        {
            __device__ auto operator()(AbsUniformCudaHipBuiltIn const& /* abs_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::fabsf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::fabs(arg);
                else if constexpr(is_decayed_v<TArg, int>)
                    return ::abs(arg);
                else if constexpr(is_decayed_v<TArg, long int>)
                    return ::labs(arg);
                else if constexpr(is_decayed_v<TArg, long long int>)
                    return ::llabs(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA acos trait specialization.
        template<typename TArg>
        struct Acos<AcosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(AcosUniformCudaHipBuiltIn const& /* acos_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::acosf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::acos(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA asin trait specialization.
        template<typename TArg>
        struct Asin<AsinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(AsinUniformCudaHipBuiltIn const& /* asin_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::asinf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::asin(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA atan trait specialization.
        template<typename TArg>
        struct Atan<AtanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(AtanUniformCudaHipBuiltIn const& /* atan_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::atanf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::atan(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA atan2 trait specialization.
        template<typename Ty, typename Tx>
        struct Atan2<
            Atan2UniformCudaHipBuiltIn,
            Ty,
            Tx,
            std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
        {
            __device__ auto operator()(Atan2UniformCudaHipBuiltIn const& /* atan2_ctx */, Ty const& y, Tx const& x)
            {
                if constexpr(is_decayed_v<Ty, float> && is_decayed_v<Tx, float>)
                    return ::atan2f(y, x);
                else if constexpr(is_decayed_v<Ty, double> || is_decayed_v<Tx, double>)
                    return ::atan2(y, x);
                else
                    static_assert(!sizeof(Ty), "Unsupported data type");

                ALPAKA_UNREACHABLE(Ty{});
            }
        };

        //! The CUDA cbrt trait specialization.
        template<typename TArg>
        struct Cbrt<CbrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            __device__ auto operator()(CbrtUniformCudaHipBuiltIn const& /* cbrt_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::cbrtf(arg);
                else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                    return ::cbrt(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA ceil trait specialization.
        template<typename TArg>
        struct Ceil<CeilUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(CeilUniformCudaHipBuiltIn const& /* ceil_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::ceilf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::ceil(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA cos trait specialization.
        template<typename TArg>
        struct Cos<CosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(CosUniformCudaHipBuiltIn const& /* cos_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::cosf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::cos(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA erf trait specialization.
        template<typename TArg>
        struct Erf<ErfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(ErfUniformCudaHipBuiltIn const& /* erf_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::erff(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::erf(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA exp trait specialization.
        template<typename TArg>
        struct Exp<ExpUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(ExpUniformCudaHipBuiltIn const& /* exp_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::expf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::exp(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA floor trait specialization.
        template<typename TArg>
        struct Floor<FloorUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(FloorUniformCudaHipBuiltIn const& /* floor_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::floorf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::floor(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA fmod trait specialization.
        template<typename Tx, typename Ty>
        struct Fmod<
            FmodUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
        {
            __device__ auto operator()(FmodUniformCudaHipBuiltIn const& /* fmod_ctx */, Tx const& x, Ty const& y)
            {
                if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::fmodf(x, y);
                else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                    return ::fmod(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]]
                = std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA isfinite trait specialization.
        template<typename TArg>
        struct Isfinite<IsfiniteUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(IsfiniteUniformCudaHipBuiltIn const& /* ctx */, TArg const& arg)
            {
                return ::isfinite(arg);
            }
        };

        //! The CUDA isinf trait specialization.
        template<typename TArg>
        struct Isinf<IsinfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(IsinfUniformCudaHipBuiltIn const& /* ctx */, TArg const& arg)
            {
                return ::isinf(arg);
            }
        };

        //! The CUDA isnan trait specialization.
        template<typename TArg>
        struct Isnan<IsnanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(IsnanUniformCudaHipBuiltIn const& /* ctx */, TArg const& arg)
            {
                return ::isnan(arg);
            }
        };

        //! The CUDA log trait specialization.
        template<typename TArg>
        struct Log<LogUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(LogUniformCudaHipBuiltIn const& /* log_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::logf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::log(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA max trait specialization.
        template<typename Tx, typename Ty>
        struct Max<
            MaxUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            __device__ auto operator()(MaxUniformCudaHipBuiltIn const& /* max_ctx */, Tx const& x, Ty const& y)
            {
                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return ::max(x, y);
                else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::fmaxf(x, y);
                else if constexpr(
                    is_decayed_v<
                        Tx,
                        double> || is_decayed_v<Ty, double> || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
                    || (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
                    return ::fmax(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]] = std::conditional_t<
                    std::is_integral_v<Tx> && std::is_integral_v<Ty>,
                    decltype(::max(x, y)),
                    std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA min trait specialization.
        template<typename Tx, typename Ty>
        struct Min<
            MinUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            __device__ auto operator()(MinUniformCudaHipBuiltIn const& /* min_ctx */, Tx const& x, Ty const& y)
            {
                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return ::min(x, y);
                else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::fminf(x, y);
                else if constexpr(
                    is_decayed_v<
                        Tx,
                        double> || is_decayed_v<Ty, double> || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
                    || (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
                    return ::fmin(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]] = std::conditional_t<
                    std::is_integral_v<Tx> && std::is_integral_v<Ty>,
                    decltype(::min(x, y)),
                    std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA pow trait specialization.
        template<typename TBase, typename TExp>
        struct Pow<
            PowUniformCudaHipBuiltIn,
            TBase,
            TExp,
            std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
        {
            __device__ auto operator()(
                PowUniformCudaHipBuiltIn const& /* pow_ctx */,
                TBase const& base,
                TExp const& exp)
            {
                if constexpr(is_decayed_v<TBase, float> && is_decayed_v<TExp, float>)
                    return ::powf(base, exp);
                else if constexpr(is_decayed_v<TBase, double> || is_decayed_v<TExp, double>)
                    return ::pow(base, exp);
                else
                    static_assert(!sizeof(TBase), "Unsupported data type");

                using Ret [[maybe_unused]]
                = std::conditional_t<is_decayed_v<TBase, float> && is_decayed_v<TExp, float>, float, double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA remainder trait specialization.
        template<typename Tx, typename Ty>
        struct Remainder<
            RemainderUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
        {
            __device__ auto operator()(
                RemainderUniformCudaHipBuiltIn const& /* remainder_ctx */,
                Tx const& x,
                Ty const& y)
            {
                if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::remainderf(x, y);
                else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                    return ::remainder(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]]
                = std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };


        //! The CUDA round trait specialization.
        template<typename TArg>
        struct Round<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(RoundUniformCudaHipBuiltIn const& /* round_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::roundf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::round(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA lround trait specialization.
        template<typename TArg>
        struct Lround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(RoundUniformCudaHipBuiltIn const& /* lround_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::lroundf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::lround(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(long{});
            }
        };

        //! The CUDA llround trait specialization.
        template<typename TArg>
        struct Llround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(RoundUniformCudaHipBuiltIn const& /* llround_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::llroundf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::llround(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                // NVCC versions before 11.3 are unable to compile 'long long{}': "type name is not allowed".
                using Ret [[maybe_unused]] = long long;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA rsqrt trait specialization.
        template<typename TArg>
        struct Rsqrt<RsqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            __device__ auto operator()(RsqrtUniformCudaHipBuiltIn const& /* rsqrt_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::rsqrtf(arg);
                else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                    return ::rsqrt(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA sin trait specialization.
        template<typename TArg>
        struct Sin<SinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(SinUniformCudaHipBuiltIn const& /* sin_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::sinf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::sin(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA sincos trait specialization.
        template<typename TArg>
        struct SinCos<SinCosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(
                SinCosUniformCudaHipBuiltIn const& /* sincos_ctx */,
                TArg const& arg,
                TArg& result_sin,
                TArg& result_cos) -> void
            {
                if constexpr(is_decayed_v<TArg, float>)
                    ::sincosf(arg, &result_sin, &result_cos);
                else if constexpr(is_decayed_v<TArg, double>)
                    ::sincos(arg, &result_sin, &result_cos);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");
            }
        };

        //! The CUDA sqrt trait specialization.
        template<typename TArg>
        struct Sqrt<SqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            __device__ auto operator()(SqrtUniformCudaHipBuiltIn const& /* sqrt_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::sqrtf(arg);
                else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                    return ::sqrt(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA tan trait specialization.
        template<typename TArg>
        struct Tan<TanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(TanUniformCudaHipBuiltIn const& /* tan_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::tanf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::tan(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA trunc trait specialization.
        template<typename TArg>
        struct Trunc<TruncUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __device__ auto operator()(TruncUniformCudaHipBuiltIn const& /* trunc_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::truncf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::trunc(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };
    } // namespace trait
#    endif
} // namespace alpaka::math

#endif
