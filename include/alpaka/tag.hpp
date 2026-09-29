/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/unused.hpp"

#include <cassert>
#include <string>
#include <tuple>

namespace alpaka
{
    namespace category
    {
        /** Category tag for queue-kind policies. */
        struct QueueKind
        {
            template<typename T_Policy>
            constexpr bool operator()(T_Policy) const;
        };

        /** Category tag for timing policies. */
        struct Timing
        {
            template<typename T_Policy>
            constexpr bool operator()(T_Policy) const;
        };
    } // namespace category

    namespace object
    {
        struct Api
        {
        };

        constexpr Api api;

        struct DeviceKind
        {
        };

        constexpr DeviceKind deviceKind;

        ALPAKA_TAG(exec);

        ALPAKA_TAG(launchedWidthFrameSpec);

        ALPAKA_TAG(dynSharedMemBytes);

        struct WarpSize
        {
        };

        constexpr WarpSize warpSize;
    } // namespace object

    namespace queueKind
    {
        namespace trait
        {
            template<typename T_QueueKind>
            struct IsQueueKind : std::is_base_of<category::QueueKind, T_QueueKind>
            {
            };
        } // namespace trait

        template<typename T_QueueKind>
        constexpr bool isQueueKind_v = trait::IsQueueKind<T_QueueKind>::value;
    } // namespace queueKind

    namespace concepts
    {
        /** Concept to check if a type is a queue kind
         *
         * @details
         * Example queue kinds are alpaka::queueKind::Blocking or alpaka::queueKind::NonBlocking.
         */
        template<typename T_QueueKind>
        concept QueueKind = queueKind::isQueueKind_v<T_QueueKind>;
    } // namespace concepts

    namespace queueKind
    {
        constexpr bool operator==(alpaka::concepts::QueueKind auto lhs, alpaka::concepts::QueueKind auto rhs)
        {
            return std::is_same_v<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>;
        }

        constexpr bool operator!=(alpaka::concepts::QueueKind auto lhs, alpaka::concepts::QueueKind auto rhs)
        {
            return !(lhs == rhs);
        }

        /** Queue should block during the task execution
         */
        struct Blocking : category::QueueKind
        {
            static constexpr bool isBlocking()
            {
                return true;
            }

            static std::string getName()
            {
                return "Blocking";
            }
        };

        constexpr auto blocking = Blocking{};

        /** Queue should process task asynchronously
         */
        struct NonBlocking : category::QueueKind
        {
            static constexpr bool isBlocking()
            {
                return false;
            }

            static std::string getName()
            {
                return "NonBlocking";
            }
        };

        constexpr auto nonBlocking = NonBlocking{};
    } // namespace queueKind

    namespace timing
    {
        namespace trait
        {
            template<typename T_Timing>
            struct IsTiming : std::is_base_of<category::Timing, T_Timing>
            {
            };
        } // namespace trait

        template<typename T_Timing>
        constexpr bool isTiming_v = trait::IsTiming<T_Timing>::value;
    } // namespace timing

    namespace concepts
    {
        /** Concept to check if a type selects whether timing is enabled. */
        template<typename T_Timing>
        concept Timing = timing::isTiming_v<T_Timing>;
    } // namespace concepts

    namespace timing
    {
        constexpr bool operator==(alpaka::concepts::Timing auto lhs, alpaka::concepts::Timing auto rhs)
        {
            return std::is_same_v<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>;
        }

        constexpr bool operator!=(alpaka::concepts::Timing auto lhs, alpaka::concepts::Timing auto rhs)
        {
            return !(lhs == rhs);
        }

        /** Backend timing information is available. */
        struct Enabled : category::Timing
        {
            static std::string getName()
            {
                return "Enabled";
            }
        };

        constexpr auto enabled = Enabled{};

        /** Backend timing information is not requested. */
        struct Disabled : category::Timing
        {
            static std::string getName()
            {
                return "Disabled";
            }
        };

        constexpr auto disabled = Disabled{};
    } // namespace timing

    namespace deviceKind
    {
        namespace detail
        {
            struct DeviceKindBase
            {
            };
        } // namespace detail

        namespace trait
        {
            template<typename T_DeviceKind>
            struct IsDeviceKind : std::is_base_of<detail::DeviceKindBase, T_DeviceKind>
            {
            };
        } // namespace trait

        template<typename T_DeviceKind>
        constexpr bool isDeviceKind_v = trait::IsDeviceKind<T_DeviceKind>::value;
    } // namespace deviceKind

    namespace concepts
    {
        /** @brief Concept to check if something is a device kind
         *
         * @details
         * A device kind in alpaka is a type of acceleration device, such as a GPU vendor. Examples are
         * alpaka::deviceKind::amdGpu or alpaka::deviceKind::cpu. Together with an alpaka::onHost::Api, it can make
         * up an alpaka::onHost::Device.
         */
        template<typename T_DeviceKind>
        concept DeviceKind = deviceKind::isDeviceKind_v<T_DeviceKind>;
    } // namespace concepts

    namespace deviceKind
    {
        constexpr bool operator==(concepts::DeviceKind auto lhs, concepts::DeviceKind auto rhs)
        {
            return std::is_same_v<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>;
        }

        constexpr bool operator!=(concepts::DeviceKind auto lhs, concepts::DeviceKind auto rhs)
        {
            return !(lhs == rhs);
        }

        struct Cpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "Cpu";
            }
        };

        constexpr auto cpu = Cpu{};

        struct NumaCpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "NumaCpu";
            }
        };

        constexpr auto numaCpu = NumaCpu{};

        struct AmdGpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "AmdGpu";
            }
        };

        constexpr auto amdGpu = AmdGpu{};

        struct NvidiaGpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "NvidiaGpu";
            }
        };

        constexpr auto nvidiaGpu = NvidiaGpu{};

        struct IntelGpu : detail::DeviceKindBase
        {
            static std::string getName()
            {
                return "IntelGpu";
            }
        };

        constexpr auto intelGpu = IntelGpu{};

        constexpr auto allDevices = std::make_tuple(cpu, numaCpu, amdGpu, nvidiaGpu, intelGpu);

    } // namespace deviceKind

    namespace layer
    {
        namespace detail
        {
            struct LayerBase
            {
            };
        } // namespace detail

        namespace trait
        {
            template<typename T_Layer>
            struct IsLayer : std::is_base_of<detail::LayerBase, T_Layer>
            {
            };
        } // namespace trait

        template<typename T_Layer>
        constexpr bool isLayer_v = trait::IsLayer<T_Layer>::value;
    } // namespace layer

    namespace concepts
    {
        /** @brief Concept to check for a compute layer of an accelerator
         *
         * @details
         * A layer is one specific part of the compute hierarchy of accelerators, for example alpaka::layer::Thread or
         * alpaka::layer::Block.
         */
        template<typename T_Layer>
        concept Layer = layer::isLayer_v<T_Layer>;
    } // namespace concepts

    namespace layer
    {
        struct Thread : detail::LayerBase
        {
        };

        constexpr auto thread = Thread{};

        struct Block : detail::LayerBase
        {
        };

        constexpr auto block = Block{};

        ALPAKA_TAG(shared);
        ALPAKA_TAG(dynShared);
    } // namespace layer

    namespace action
    {
        ALPAKA_TAG(threadBlockSync);
    } // namespace action

    struct Empty
    {
    };

    namespace exec
    {
        namespace trait
        {
            template<typename T_Executor>
            struct IsSeqExecutor : std::false_type
            {
            };
        } // namespace trait

        template<typename T_Exec>
        constexpr bool isSeqExecutor_v = trait::IsSeqExecutor<T_Exec>::value;
    } // namespace exec

    /** check if an executor can only be used with a single thread per block
     *
     * @return true if a block can only have a single thread, else false
     */
    template<typename T_Exec>
    consteval bool isSeqExecutor(T_Exec exec)
    {
        alpaka::unused(exec);
        return exec::isSeqExecutor_v<T_Exec>;
    }

    namespace category
    {
        template<typename T_Policy>
        constexpr bool QueueKind::operator()(T_Policy) const
        {
            return alpaka::concepts::QueueKind<T_Policy>;
        }

        template<typename T_Policy>
        constexpr bool Timing::operator()(T_Policy) const
        {
            return alpaka::concepts::Timing<T_Policy>;
        }
    } // namespace category
} // namespace alpaka
