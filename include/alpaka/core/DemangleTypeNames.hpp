/* Copyright 2022 Andrea Bocci, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <string>
#include <typeinfo>

#if defined(ALPAKA_HAS_BOOST_HEADERS)
#    include <boost/core/demangle.hpp>
#endif

namespace alpaka::core
{
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#    pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#endif
#if defined(ALPAKA_HAS_BOOST_HEADERS)
    template<typename T>
    inline std::string const demangled = boost::core::demangle(typeid(T).name());
#else
    template<typename T>
    inline std::string const demangled = typeid(T).name();
#endif
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif

} // namespace alpaka::core
