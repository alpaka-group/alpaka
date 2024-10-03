/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Decay.hpp"
#include "alpaka/math/Traits.hpp"

namespace alpaka::math
{
    //! The standard library abs, implementation covered by the general template.
    class AbsStdLib : public interface::Implements<ConceptMathAbs, AbsStdLib>
    {
    };

    //! The standard library acos, implementation covered by the general template.
    class AcosStdLib : public interface::Implements<ConceptMathAcos, AcosStdLib>
    {
    };

    //! The standard library acos, implementation covered by the general template.
    class AcoshStdLib : public interface::Implements<ConceptMathAcosh, AcoshStdLib>
    {
    };

    //! The standard library arg, implementation covered by the general template.
    class ArgStdLib : public interface::Implements<ConceptMathArg, ArgStdLib>
    {
    };

    //! The standard library asin, implementation covered by the general template.
    class AsinStdLib : public interface::Implements<ConceptMathAsin, AsinStdLib>
    {
    };

    //! The standard library asinh, implementation covered by the general template.
    class AsinhStdLib : public interface::Implements<ConceptMathAsinh, AsinhStdLib>
    {
    };

    //! The standard library atan, implementation covered by the general template.
    class AtanStdLib : public interface::Implements<ConceptMathAtan, AtanStdLib>
    {
    };

    //! The standard library atanh, implementation covered by the general template.
    class AtanhStdLib : public interface::Implements<ConceptMathAtanh, AtanhStdLib>
    {
    };

    //! The standard library atan2, implementation covered by the general template.
    class Atan2StdLib : public interface::Implements<ConceptMathAtan2, Atan2StdLib>
    {
    };

    //! The standard library cbrt, implementation covered by the general template.
    class CbrtStdLib : public interface::Implements<ConceptMathCbrt, CbrtStdLib>
    {
    };

    //! The standard library ceil, implementation covered by the general template.
    class CeilStdLib : public interface::Implements<ConceptMathCeil, CeilStdLib>
    {
    };

    //! The standard library conj, implementation covered by the general template.
    class ConjStdLib : public interface::Implements<ConceptMathConj, ConjStdLib>
    {
    };

    //! The standard library copysign, implementation covered by the general template.
    class CopysignStdLib : public interface::Implements<ConceptMathCopysign, CopysignStdLib>
    {
    };

    //! The standard library cos, implementation covered by the general template.
    class CosStdLib : public interface::Implements<ConceptMathCos, CosStdLib>
    {
    };

    //! The standard library cosh, implementation covered by the general template.
    class CoshStdLib : public interface::Implements<ConceptMathCosh, CoshStdLib>
    {
    };

    //! The standard library erf, implementation covered by the general template.
    class ErfStdLib : public interface::Implements<ConceptMathErf, ErfStdLib>
    {
    };

    //! The standard library exp, implementation covered by the general template.
    class ExpStdLib : public interface::Implements<ConceptMathExp, ExpStdLib>
    {
    };

    //! The standard library floor, implementation covered by the general template.
    class FloorStdLib : public interface::Implements<ConceptMathFloor, FloorStdLib>
    {
    };

    //! The standard library fma, implementation covered by the general template.
    class FmaStdLib : public interface::Implements<ConceptMathFma, FmaStdLib>
    {
    };

    //! The standard library fmod, implementation covered by the general template.
    class FmodStdLib : public interface::Implements<ConceptMathFmod, FmodStdLib>
    {
    };

    //! The standard library isfinite, implementation covered by the general template.
    class IsfiniteStdLib : public interface::Implements<ConceptMathIsfinite, IsfiniteStdLib>
    {
    };

    //! The standard library isinf, implementation covered by the general template.
    class IsinfStdLib : public interface::Implements<ConceptMathIsinf, IsinfStdLib>
    {
    };

    //! The standard library isnan, implementation covered by the general template.
    class IsnanStdLib : public interface::Implements<ConceptMathIsnan, IsnanStdLib>
    {
    };

    //! The standard library log, implementation covered by the general template.
    class LogStdLib : public interface::Implements<ConceptMathLog, LogStdLib>
    {
    };

    //! The standard library log2, implementation covered by the general template.
    class Log2StdLib : public interface::Implements<ConceptMathLog2, Log2StdLib>
    {
    };

    //! The standard library log10, implementation covered by the general template.
    class Log10StdLib : public interface::Implements<ConceptMathLog10, Log10StdLib>
    {
    };

    //! The standard library max.
    class MaxStdLib : public interface::Implements<ConceptMathMax, MaxStdLib>
    {
    };

    //! The standard library min.
    class MinStdLib : public interface::Implements<ConceptMathMin, MinStdLib>
    {
    };

    //! The standard library pow, implementation covered by the general template.
    class PowStdLib : public interface::Implements<ConceptMathPow, PowStdLib>
    {
    };

    //! The standard library remainder, implementation covered by the general template.
    class RemainderStdLib : public interface::Implements<ConceptMathRemainder, RemainderStdLib>
    {
    };

    //! The standard library round, implementation covered by the general template.
    class RoundStdLib : public interface::Implements<ConceptMathRound, RoundStdLib>
    {
    };

    //! The standard library rsqrt, implementation covered by the general template.
    class RsqrtStdLib : public interface::Implements<ConceptMathRsqrt, RsqrtStdLib>
    {
    };

    //! The standard library sin, implementation covered by the general template.
    class SinStdLib : public interface::Implements<ConceptMathSin, SinStdLib>
    {
    };

    //! The standard library sinh, implementation covered by the general template.
    class SinhStdLib : public interface::Implements<ConceptMathSinh, SinhStdLib>
    {
    };

    //! The standard library sincos, implementation covered by the general template.
    class SinCosStdLib : public interface::Implements<ConceptMathSinCos, SinCosStdLib>
    {
    };

    //! The standard library sqrt, implementation covered by the general template.
    class SqrtStdLib : public interface::Implements<ConceptMathSqrt, SqrtStdLib>
    {
    };

    //! The standard library tan, implementation covered by the general template.
    class TanStdLib : public interface::Implements<ConceptMathTan, TanStdLib>
    {
    };

    //! The standard library tanh, implementation covered by the general template.
    class TanhStdLib : public interface::Implements<ConceptMathTanh, TanhStdLib>
    {
    };

    //! The standard library trunc, implementation covered by the general template.
    class TruncStdLib : public interface::Implements<ConceptMathTrunc, TruncStdLib>
    {
    };

    //! The standard library math trait specializations.
    class MathStdLib
        : public AbsStdLib
        , public AcosStdLib
        , public AcoshStdLib
        , public ArgStdLib
        , public AsinStdLib
        , public AsinhStdLib
        , public AtanStdLib
        , public AtanhStdLib
        , public Atan2StdLib
        , public CbrtStdLib
        , public CeilStdLib
        , public ConjStdLib
        , public CopysignStdLib
        , public CosStdLib
        , public CoshStdLib
        , public ErfStdLib
        , public ExpStdLib
        , public FloorStdLib
        , public FmaStdLib
        , public FmodStdLib
        , public LogStdLib
        , public Log2StdLib
        , public Log10StdLib
        , public MaxStdLib
        , public MinStdLib
        , public PowStdLib
        , public RemainderStdLib
        , public RoundStdLib
        , public RsqrtStdLib
        , public SinStdLib
        , public SinhStdLib
        , public SinCosStdLib
        , public SqrtStdLib
        , public TanStdLib
        , public TanhStdLib
        , public TruncStdLib
        , public IsnanStdLib
        , public IsinfStdLib
        , public IsfiniteStdLib
    {
    };

    namespace trait
    {
        //! The standard library max trait specialization.
        template<typename Tx, typename Ty>
        struct Max<MaxStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            ALPAKA_FN_HOST auto operator()(MaxStdLib const& /* max_ctx */, Tx const& x, Ty const& y)
            {
                using std::fmax;
                using std::max;

                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return max(x, y);
                else if constexpr(
                    is_decayed_v<Tx, float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double>
                    || is_decayed_v<Ty, double>)
                    return fmax(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
            }
        };

        //! The standard library min trait specialization.
        template<typename Tx, typename Ty>
        struct Min<MinStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            ALPAKA_FN_HOST auto operator()(MinStdLib const& /* min_ctx */, Tx const& x, Ty const& y)
            {
                using std::fmin;
                using std::min;

                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return min(x, y);
                else if constexpr(
                    is_decayed_v<Tx, float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double>
                    || is_decayed_v<Ty, double>)
                    return fmin(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
            }
        };
    } // namespace trait

} // namespace alpaka::math
