Warp and Subgroup Functions
===========================

A warp is a hardware-scheduled group of threads that share a common execution context and execute instructions collectively,
while individual threads may be active or inactive (masked) due to control-flow divergence.
Threads in a warp can exchange values via warp shuffle functions without going through :ref:`shared memory <shared-memory>`.
A thread block may contain multiple warps.
The number of threads within a thread block is not required to be a multiple of the warp size.
Threads in different warps cannot use warp shuffle functions to exchange values.
A warp is always a one-dimensional group of threads, even within n-dimensional kernels.

When to Reach for Warp Functions
--------------------------------

Use warp functions when:

- you want fast communication among threads that execute in lock-step
- you are implementing a reduction or prefix-style pattern inside a warp
- you need ballot-style voting or lane-to-lane value exchange.

A Warp Reduction With ``shflDown``
----------------------------------

The following example reduces one value per lane to one value per warp.
``makeIdxMap`` uses warps directly instead of first mapping at the thread-block level.

  .. literalinclude:: ../../snippets/example/190_warp.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-warpKernel
    :end-before: END-TUTORIAL-warpKernel
    :dedent:

Important rules:

- All participating threads must call the same warp intrinsic in a compatible control-flow region.
- Use the actual `warp size reported by the accelerator <https://alpaka.readthedocs.io/en/latest/doxygen/structalpaka_1_1onAcc_1_1Acc.html>`_ instead of hard-coding ``32``, which is typical for NVIDIA devices.
- On host devices, the warp size can be ``1``. The code still compiles and runs, but the subgroup behavior is naturally trivial there.

`Other warp <../doxygen/namespacealpaka_1_1onAcc_1_1warp.html>`_ functions:

- ``onAcc::warp::shfl`` to broadcast from a chosen lane
- ``onAcc::warp::shflUp`` read from the lower lane
- ``onAcc::warp::shflXor`` ``xor`` the read value from a lane with its own
- ``onAcc::warp::all`` and ``onAcc::warp::any`` for voting between participating warp threads
- ``onAcc::warp::ballot`` for predicate masks

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>190_warp.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/190_warp.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
