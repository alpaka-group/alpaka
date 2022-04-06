/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Decay.hpp>
#include <alpaka/math/Traits.hpp>

namespace alpaka::math
{
    //! The standard library abs, implementation covered by the general template.
    class AbsStdLib : public concepts::Implements<ConceptMathAbs, AbsStdLib>
    {
    };

    //! The standard library acos, implementation covered by the general template.
    class AcosStdLib : public concepts::Implements<ConceptMathAcos, AcosStdLib>
    {
    };

    //! The standard library arg, implementation covered by the general template.
    class ArgStdLib : public concepts::Implements<ConceptMathArg, ArgStdLib>
    {
    };

    //! The standard library asin, implementation covered by the general template.
    class AsinStdLib : public concepts::Implements<ConceptMathAsin, AsinStdLib>
    {
    };

    //! The standard library atan, implementation covered by the general template.
    class AtanStdLib : public concepts::Implements<ConceptMathAtan, AtanStdLib>
    {
    };

    //! The standard library atan2, implementation covered by the general template.
    class Atan2StdLib : public concepts::Implements<ConceptMathAtan2, Atan2StdLib>
    {
    };

    //! The standard library cbrt, implementation covered by the general template.
    class CbrtStdLib : public concepts::Implements<ConceptMathCbrt, CbrtStdLib>
    {
    };

    //! The standard library ceil, implementation covered by the general template.
    class CeilStdLib : public concepts::Implements<ConceptMathCeil, CeilStdLib>
    {
    };

    //! The standard library conj, implementation covered by the general template.
    class ConjStdLib : public concepts::Implements<ConceptMathConj, ConjStdLib>
    {
    };

    //! The standard library cos, implementation covered by the general template.
    class CosStdLib : public concepts::Implements<ConceptMathCos, CosStdLib>
    {
    };

    //! The standard library erf, implementation covered by the general template.
    class ErfStdLib : public concepts::Implements<ConceptMathErf, ErfStdLib>
    {
    };

    //! The standard library exp, implementation covered by the general template.
    class ExpStdLib : public concepts::Implements<ConceptMathExp, ExpStdLib>
    {
    };

    //! The standard library floor, implementation covered by the general template.
    class FloorStdLib : public concepts::Implements<ConceptMathFloor, FloorStdLib>
    {
    };

    //! The standard library fmod, implementation covered by the general template.
    class FmodStdLib : public concepts::Implements<ConceptMathFmod, FmodStdLib>
    {
    };

    //! The standard library isfinite, implementation covered by the general template.
    class IsfiniteStdLib : public concepts::Implements<ConceptMathIsfinite, IsfiniteStdLib>
    {
    };

    //! The standard library isinf, implementation covered by the general template.
    class IsinfStdLib : public concepts::Implements<ConceptMathIsinf, IsinfStdLib>
    {
    };

    //! The standard library isnan, implementation covered by the general template.
    class IsnanStdLib : public concepts::Implements<ConceptMathIsnan, IsnanStdLib>
    {
    };

    //! The standard library log, implementation covered by the general template.
    class LogStdLib : public concepts::Implements<ConceptMathLog, LogStdLib>
    {
    };

    //! The standard library max.
    class MaxStdLib : public concepts::Implements<ConceptMathMax, MaxStdLib>
    {
    };

    //! The standard library min.
    class MinStdLib : public concepts::Implements<ConceptMathMin, MinStdLib>
    {
    };

    //! The standard library pow, implementation covered by the general template.
    class PowStdLib : public concepts::Implements<ConceptMathPow, PowStdLib>
    {
    };

    //! The standard library remainder, implementation covered by the general template.
    class RemainderStdLib : public concepts::Implements<ConceptMathRemainder, RemainderStdLib>
    {
    };

    //! The standard library round, implementation covered by the general template.
    class RoundStdLib : public concepts::Implements<ConceptMathRound, RoundStdLib>
    {
    };

    //! The standard library rsqrt, implementation covered by the general template.
    class RsqrtStdLib : public concepts::Implements<ConceptMathRsqrt, RsqrtStdLib>
    {
    };

    //! The standard library sin, implementation covered by the general template.
    class SinStdLib : public concepts::Implements<ConceptMathSin, SinStdLib>
    {
    };

    //! The standard library sincos, implementation covered by the general template.
    class SinCosStdLib : public concepts::Implements<ConceptMathSinCos, SinCosStdLib>
    {
    };

    //! The standard library sqrt, implementation covered by the general template.
    class SqrtStdLib : public concepts::Implements<ConceptMathSqrt, SqrtStdLib>
    {
    };

    //! The standard library tan, implementation covered by the general template.
    class TanStdLib : public concepts::Implements<ConceptMathTan, TanStdLib>
    {
    };

    //! The standard library trunc, implementation covered by the general template.
    class TruncStdLib : public concepts::Implements<ConceptMathTrunc, TruncStdLib>
    {
    };

    //! The standard library math trait specializations.
    class MathStdLib
        : public AbsStdLib
        , public AcosStdLib
        , public ArgStdLib
        , public AsinStdLib
        , public AtanStdLib
        , public Atan2StdLib
        , public CbrtStdLib
        , public CeilStdLib
        , public ConjStdLib
        , public CosStdLib
        , public ErfStdLib
        , public ExpStdLib
        , public FloorStdLib
        , public FmodStdLib
        , public LogStdLib
        , public MaxStdLib
        , public MinStdLib
        , public PowStdLib
        , public RemainderStdLib
        , public RoundStdLib
        , public RsqrtStdLib
        , public SinStdLib
        , public SinCosStdLib
        , public SqrtStdLib
        , public TanStdLib
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
                    is_decayed_v<
                        Tx,
                        float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
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
                    is_decayed_v<
                        Tx,
                        float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                    return fmin(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
            }
        };
    } // namespace trait

} // namespace alpaka::math
