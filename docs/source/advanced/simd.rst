SIMD Guide
==========

Goal
----

Provide portable SIMD operations on CPU (vectorized) and GPU (scalar, width=1) with unified API.

Required Build Flags
--------------------

Always use:

* ``-O3`` (Release build)
* ``-march=native`` (CPU-specific optimizations)

Mixed GPU + CPU Builds
----------------------

Forward CPU flags to device compiler:

* **CUDA**: ``-DCMAKE_CUDA_FLAGS="-Xcompiler -march=native"``
* **HIP**: ``-DCMAKE_HIP_FLAGS="-Xclang -march=native"``

Without this, host code under NVCC/HIP downgrades to generic ISA.

Quick Build Commands
--------------------

**CPU only:**

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"

**CUDA+CPU:**

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_CUDA_FLAGS="-Xcompiler -march=native"

Policy Selection
----------------

* **std::experimental::simd**: Pure CPU builds (best performance)
* **Fallback**: Mixed builds or when experimental SIMD unavailable

Verification
------------

Check example output shows:

- SIMD width > 1 on CPU
- Correct policy name
- Expected speedup

Troubleshooting
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Issue
     - Fix
   * - Width=1 on CPU
     - Add ``-march=native``
   * - Poor speedup
     - Ensure ``-O3`` and Release build
   * - Fallback used
     - Expected in mixed builds

Pack Iteration Helper
---------------------

Examples use ``alpaka::simd::SimdAlgo::forEach`` for automatic:

- Full SIMD pack iteration
- Tail handling
- No manual remainder code needed

Explicit x86-64 microarchitecture flags (alternative to ``-march=native``)
--------------------------------------------------------------------------

Use explicit levels only if you are certain the deployment machines provide the required ISA.

Baseline psABI levels (GCC / Clang):

* ``-march=x86-64-v2`` – adds SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16
* ``-march=x86-64-v3`` – v2 + AVX, AVX2, FMA, BMI1, BMI2, MOVBE
* ``-march=x86-64-v4`` – v3 + AVX-512F (typically also BW, DQ, CD, VL)

Common compiler / OS variants (approximate mapping):

* GCC / Clang: ``-march=x86-64-v2`` / ``-march=x86-64-v3`` / ``-march=x86-64-v4``
* Intel (icx / icc): ``-march=core2`` (≈v2), ``-march=core-avx2`` (≈v3), ``-march=skylake-avx512`` (≈v4)
* MSVC: ``/arch:AVX2`` (≈v3), ``/arch:AVX512`` (≈v4, if supported)
* Apple Clang (macOS): ``-march=core2`` (≈v2), ``-march=core-avx2`` (≈v3), no v4 support
