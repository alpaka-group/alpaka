/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/extent/Traits.hpp"

namespace alpaka
{
    struct ConceptMemAlloc
    {
    };

    //! The allocator traits.
    namespace trait
    {
        //! The memory allocation trait.
        template<typename T, typename TAlloc, typename TSfinae = void>
        struct Malloc;

        //! The memory free trait.
        template<typename T, typename TAlloc, typename TSfinae = void>
        struct Free;
    } // namespace trait

    //! \return The pointer to the allocated memory.
    template<typename T, typename TAlloc>
    ALPAKA_FN_HOST auto malloc(TAlloc const& alloc, std::size_t const& sizeElems) -> T*
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMemAlloc, TAlloc>;
        return trait::Malloc<T, ImplementationBase>::malloc(alloc, sizeElems);
    }

    //! Frees the memory identified by the given pointer.
    template<typename TAlloc, typename T>
    ALPAKA_FN_HOST auto free(TAlloc const& alloc, T const* const ptr) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMemAlloc, TAlloc>;
        trait::Free<T, ImplementationBase>::free(alloc, ptr);
    }
} // namespace alpaka
