/* Copyright 2019 Benjamin Worpitz
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/alpaka.hpp"

#include <tuple>

namespace alpaka::test
{
    template<typename TDevQueue>
    struct QueueTestFixture
    {
        using Dev = std::tuple_element_t<0, TDevQueue>;
        using Queue = std::tuple_element_t<1, TDevQueue>;

        using Pltf = alpaka::Pltf<Dev>;

        QueueTestFixture() : m_dev(getDevByIdx<Pltf>(0u)), m_queue(m_dev)
        {
        }

        Dev m_dev;
        Queue m_queue;
    };
} // namespace alpaka::test
