/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/queue/Traits.hpp"

#include <type_traits>
#include <vector>

namespace alpaka
{
    struct ConceptPltf
    {
    };

    //! True if TPltf is a platform, i.e. if it implements the ConceptPltf concept.
    template<typename TPltf>
    inline constexpr bool isPlatform = concepts::ImplementsConcept<ConceptPltf, TPltf>::value;

    //! The platform traits.
    namespace trait
    {
        //! The platform type trait.
        template<typename T, typename TSfinae = void>
        struct PltfType;

        template<typename TPltf>
        struct PltfType<TPltf, std::enable_if_t<concepts::ImplementsConcept<ConceptPltf, TPltf>::value>>
        {
            using type = typename concepts::ImplementationBase<ConceptDev, TPltf>;
        };

        //! The device count get trait.
        template<typename T, typename TSfinae = void>
        struct GetDevCount;

        //! The device get trait.
        template<typename T, typename TSfinae = void>
        struct GetDevByIdx;
    } // namespace trait

    //! The platform type trait alias template to remove the ::type.
    template<typename T>
    using Pltf = typename trait::PltfType<T>::type;

    //! \return The device identified by its index.
    template<typename TPltf>
    ALPAKA_FN_HOST auto getDevCount(TPltf const& platform)
    {
        return trait::GetDevCount<TPltf>::getDevCount(platform);
    }

    //! \return The device identified by its index.
    template<typename TPltf>
    ALPAKA_FN_HOST auto getDevByIdx(TPltf const& platform, std::size_t const& devIdx) -> Dev<TPltf>
    {
        return trait::GetDevByIdx<TPltf>::getDevByIdx(platform, devIdx);
    }

    //! \return All the devices available on this accelerator.
    template<typename TPltf>
    ALPAKA_FN_HOST auto getDevs(TPltf const& platform) -> std::vector<Dev<TPltf>>
    {
        std::vector<Dev<TPltf>> devs;

        std::size_t const devCount = getDevCount(platform);
        devs.reserve(devCount);
        for(std::size_t devIdx(0); devIdx < devCount; ++devIdx)
        {
            devs.push_back(getDevByIdx(platform, devIdx));
        }

        return devs;
    }

    namespace trait
    {
        template<typename TPltf, typename TProperty>
        struct QueueType<TPltf, TProperty, std::enable_if_t<concepts::ImplementsConcept<ConceptPltf, TPltf>::value>>
        {
            using type = typename QueueType<typename alpaka::trait::DevType<TPltf>::type, TProperty>::type;
        };
    } // namespace trait
} // namespace alpaka
