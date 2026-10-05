/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka
{
    struct ConceptIntrinsic
    {
    };

    //! The intrinsics traits.
    namespace trait
    {
        //! The popcount trait.
        template<typename TWarp, typename TSfinae = void>
        struct Popcount;

        //! The ffs trait.
        template<typename TWarp, typename TSfinae = void>
        struct Ffs;

        //! The brev trait.
        template<typename TWarp, typename TSfinae = void>
        struct Brev;

        //! The clz trait.
        template<typename TWarp, typename TSfinae = void>
        struct Clz;

    } // namespace trait

    //! Returns the number of 1 bits in the given 32-bit value.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto popcount(TIntrinsic const& intrinsic, std::uint32_t value) -> std::int32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Popcount<ImplementationBase>::popcount(intrinsic, value);
    }

    //! Returns the number of 1 bits in the given 64-bit value.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto popcount(TIntrinsic const& intrinsic, std::uint64_t value) -> std::int32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Popcount<ImplementationBase>::popcount(intrinsic, value);
    }

    //! Returns the 1-based position of the least significant bit set to 1
    //! in the given 32-bit value. Returns 0 for input value 0.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto ffs(TIntrinsic const& intrinsic, std::int32_t value) -> std::int32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Ffs<ImplementationBase>::ffs(intrinsic, value);
    }

    //! Returns the 1-based position of the least significant bit set to 1
    //! in the given 64-bit value. Returns 0 for input value 0.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value The input value.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto ffs(TIntrinsic const& intrinsic, std::int64_t value) -> std::int32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Ffs<ImplementationBase>::ffs(intrinsic, value);
    }

    //! Reverses the bit order of a unsigned int value.
    //!
    //! This function follows the semantics of the backend-specific bit-reversal
    //! operation for std::uint32_t.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value Lane 32-bit unsigned integer.
    //! \return The bit-reversed value of \p value.

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto brev(TIntrinsic const& intrinsic, std::uint32_t value) -> std::uint32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Brev<ImplementationBase>::brev(intrinsic, value);
    }

    //! Reverses the bit order of a long unsigned int value.
    //!
    //! This function follows the semantics of the backend-specific bit-reversal
    //! operation for std::uint64_t.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value Lane 64-bit unsigned integer.
    //! \return The bit-reversed value of \p value.

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto brev(TIntrinsic const& intrinsic, std::uint64_t value) -> std::uint64_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Brev<ImplementationBase>::brev(intrinsic, value);
    }

    //! Counts the number of leading zero bits in an unsigned int.
    //!
    //! This function follows the semantics of the backend-specific count-leading-zeros
    //! operation for std::uint32_t.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value input value 32-bit unsigned integer.
    //! \return Number of leading zero bits in \p value.

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto clz(TIntrinsic const& intrinsic, std::uint32_t value) -> std::int32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Clz<ImplementationBase>::clz(intrinsic, value);
    }

    //! Counts the number of leading zero bits in a long unsigned int.
    //!
    //! This function follows the semantics of the backend-specific count-leading-zeros
    //! operation for std::uint64_t.
    //!
    //! \tparam TIntrinsic The intrinsic implementation type.
    //! \param intrinsic The intrinsic implementation.
    //! \param value input value 64-bit unsigned integer.
    //! \return Number of leading zero bits in \p value.

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIntrinsic>
    ALPAKA_FN_ACC auto clz(TIntrinsic const& intrinsic, std::uint64_t value) -> std::int32_t
    {
        using ImplementationBase = interface::ImplementationBase<ConceptIntrinsic, TIntrinsic>;
        return trait::Clz<ImplementationBase>::clz(intrinsic, value);
    }

} // namespace alpaka
