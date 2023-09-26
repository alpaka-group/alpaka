/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Antonio Di Pilato <tony.dipilato03@gmail.com>
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"

#include <type_traits>
#include <utility>

namespace alpaka::test::integ
{
    //! Measures and returns the runtime in ms of the passed callable.
    //! \param callable An object with operator().
    template<typename TCallable>
    auto measureRunTimeMs(TCallable&& callable) -> std::chrono::milliseconds::rep
    {
        auto const start = std::chrono::high_resolution_clock::now();
        std::forward<TCallable>(callable)();
        auto const end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    //! \return The run time of the given kernel.
    template<typename TQueue, typename TTask>
    auto measureTaskRunTimeMs(TQueue& queue, TTask&& task) -> std::chrono::milliseconds::rep
    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
        std::cout << "measureKernelRunTime("
                  << " queue: " << core::demangled<TQueue> << " task: " << core::demangled<std::decay_t<TTask>> << ")"
                  << std::endl;
#endif
        // Wait for the queue to finish all tasks enqueued prior to the given task.
        alpaka::wait(queue);

        return measureRunTimeMs(
            [&]
            {
                alpaka::enqueue(queue, std::forward<TTask>(task));

                // Wait for the queue to finish the task execution to measure its run time.
                alpaka::wait(queue);
            });
    }
} // namespace alpaka::test::integ
