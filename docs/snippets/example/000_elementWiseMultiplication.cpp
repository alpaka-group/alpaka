/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

// BEGIN-EXAMPLE-elementWiseMultiplication
#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

auto main() -> int
{
    unsigned n = 5;

    /* Select a device, possible combinations:
     * host+cpu, host+numaCpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
     * oneApi+amdGpu, oneApi+nvidiaGpu
     */
    auto devSelector = ap::api::host + ap::deviceKind::cpu;
    ap::onHost::Device devAcc = devSelector.makeDevice(0);
    printf("Using alpaka device: %s\n", devAcc.getName().c_str());

    // Blocking device queue (requires synchronization)
    ap::onHost::Queue queue = devAcc.makeQueue(ap::queueKind::blocking);

    // Allocate unified memory that is accessible on host and device
    auto a = ap::onHost::allocUnified<int>(devAcc, n);
    auto b = ap::onHost::allocUnified<int>(devAcc, n);
    auto c = ap::onHost::allocUnified<int>(devAcc, n);

    // Initialize values on host
    for(unsigned i = 0; i < n; i++)
    {
        a[i] = static_cast<int>(i);
        b[i] = 1;
    }

    // Run element-wise multiplication on device
    ap::onHost::transform(queue, c, std::multiplies{}, a, b);

    for(unsigned i = 0; i < n; i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}

// END-EXAMPLE-elementWiseMultiplication
