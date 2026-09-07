/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/PolicyList.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

namespace alpaka::trait
{
    /** Identifies compile-time memory-allocation policy tags.
     *
     * Specialize this trait with std::true_type to register an application-defined memory-allocation policy. Every
     * memory policy is also registered as a general policy through IsPolicy.
     *
     * @tparam T_Policy Type to inspect.
     */
    template<typename T_Policy>
    struct IsMemoryPolicy : std::false_type
    {
    };

    template<alpaka::concepts::MemoryProperty T_MemoryProperty>
    struct IsMemoryPolicy<T_MemoryProperty> : std::true_type
    {
    };
} // namespace alpaka::trait

namespace alpaka
{
    /** Whether a type is registered as a compile-time memory-allocation policy tag. */
    template<typename T_Policy>
    constexpr bool isMemoryPolicy_v = trait::IsMemoryPolicy<std::remove_cvref_t<T_Policy>>::value;

    namespace concepts
    {
        /** A registered, default-initializable compile-time memory-allocation policy tag. */
        template<typename T_Policy>
        concept MemoryPolicy = isMemoryPolicy_v<T_Policy> && std::default_initializable<std::remove_cvref_t<T_Policy>>;
    } // namespace concepts

    namespace trait
    {
        template<alpaka::concepts::MemoryPolicy T_Policy>
        struct IsPolicy<T_Policy> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

namespace alpaka::onHost
{
    /** Collection of compile-time policies used to tweak a memory allocation.
     *
     * Policies can be supplied in any order. At most one policy from each category may be present. The memory
     * property defaults to memoryProperty::defaultProperty when its policy is omitted, which leaves the placement
     * decision to the backend.
     *
     * @tparam T_Policies Memory policy tag types contained in the bundle.
     */
    template<alpaka::concepts::MemoryPolicy... T_Policies>
    struct MemoryPolicyList : PolicyList<T_Policies...>
    {
        using Base = PolicyList<T_Policies...>;

        /** Construct a memory policy bundle.
         *
         * @param policies Compile-time memory policy tags.
         */
        constexpr MemoryPolicyList(T_Policies... policies) : Base{policies...}
        {
        }

        /** Return the selected memory property, or memoryProperty::defaultProperty if none was supplied. */
        static constexpr alpaka::concepts::MemoryProperty auto getMemoryProperty()
        {
            return Base::search(category::MemoryProperty{}, memoryProperty::defaultProperty);
        }

        using Base::hasPolicy;
    };

    template<typename... T_Policies>
    MemoryPolicyList(T_Policies...) -> MemoryPolicyList<T_Policies...>;
} // namespace alpaka::onHost

namespace alpaka::concepts
{
    /** A specialization of onHost::MemoryPolicyList. */
    template<typename T_Policies>
    concept MemoryPolicyList = SpecializationOf<T_Policies, onHost::MemoryPolicyList>;
} // namespace alpaka::concepts
