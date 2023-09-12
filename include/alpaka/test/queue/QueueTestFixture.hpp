/* Copyright 2023 Benjamin Worpitz, Jan Stephan
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
#if defined(BOOST_COMP_GNUC) && BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(11, 0, 0)                                     \
    && BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(12, 0, 0)
// g++-11 (wrongly) believes that m_platform is used in an uninitialized state.
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
        using Dev = std::tuple_element_t<0, TDevQueue>;
        using Queue = std::tuple_element_t<1, TDevQueue>;
        using Platform = alpaka::Platform<Dev>;

        Platform m_platform{};
        Dev m_dev{getDevByIdx(m_platform, 0)};
        Queue m_queue{m_dev};
#if defined(BOOST_COMP_GNUC) && BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(11, 0, 0)                                     \
    && BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(12, 0, 0)
#    pragma GCC diagnostic pop
#endif
    };
} // namespace alpaka::test
