/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <string>

/**
 * @brief Provides scopes for atomic and memory fence operations, analogous to NVIDIA CUDA's atomic and fence scopes.
 *
 * This namespace defines the visibility scopes for atomic operations and memory fences,
 * which control the visibility of memory operations across threads, blocks, and devices.
 * The provided scopes are:
 * - Block: Visibility within a thread block.
 * - Device: Visibility across all thread blocks on the same device.
 * - System: System-wide visibility, mapped to the strongest available atomic/fence by the backend.
 *
 * @see alpaka::onAcc::atomicAdd, alpaka::onAcc::memFence
 */
namespace alpaka::onAcc
{
    namespace scope
    {
        /**
         * @brief Base tag for scope types.
         *
         * This tag can be used to constrain APIs that accept only valid scopes.
         */
        struct ScopeTag
        {
        };

        /**
         * @brief Scope for atomic and fence operations visible only within the same thread block.
         *
         * When used with atomic operations (e.g., atomicAdd), only threads within the same block
         * will see the updated value. When used with threadFence, it ensures that all writes
         * from the current thread are visible to all other threads in the same block.
         *
         * @note Analogous to CUDA's `atomicAdd_block` and `threadFence_block`.
         */
        struct Block : ScopeTag
        {
            static std::string getName()
            {
                return "Block";
            }
        };

        constexpr Block block{};

        /**
         * @brief Scope for atomic and fence operations visible across all thread blocks on the same device.
         *
         * When used with atomic operations, all threads on the same device will see the updated value.
         * When used with threadFence, it ensures that all writes from the current thread are visible
         * to all other threads on the same device.
         *
         * @note This scope is stronger than Block but weaker than System.
         */
        struct Device : ScopeTag
        {
            static std::string getName()
            {
                return "Device";
            }
        };

        constexpr Device device{};

        /**
         * @brief Scope for atomic and fence operations with system-wide visibility.
         *
         * When used with atomic operations, all threads in the system (potentially across multiple devices)
         * will see the updated value. When used with threadFence, it ensures that all writes from the current
         * thread are visible to all other threads in the system.
         *
         * @attention System operations are only visible to other threads of the same device kind.
         * Operations executed on a host compute device will not be visible to threads in, for example, CUDA/HIP or
         * oneAPI kernels, and vice versa.
         *
         * @note This is the strongest scope, analogous to CUDA's `atomicAdd_system` and the strongest fence.
         */
        struct System : ScopeTag
        {
            static std::string getName()
            {
                return "System";
            }
        };

        constexpr System system{};
    } // namespace scope

    namespace concepts
    {
        template<typename T>
        concept Scope = std::derived_from<T, scope::ScopeTag>;
    } // namespace concepts

} // namespace alpaka::onAcc
