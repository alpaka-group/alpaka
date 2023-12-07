/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
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
        using Platform = alpaka::Platform<Dev>;

        Platform m_platform{};
        Dev m_dev{getDevByIdx(m_platform, 0)};
        Queue m_queue{m_dev};
    };
} // namespace alpaka::test
