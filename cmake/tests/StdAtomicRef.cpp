/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 *
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <atomic>
#include <cstdlib>

#if !defined(__cpp_lib_atomic_ref)
#    error "std::atomic_ref<T> not supported!"
#endif

auto main() -> int
{
    return EXIT_SUCCESS;
}
