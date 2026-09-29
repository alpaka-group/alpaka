.. highlight:: bash

Benchmarks
==========

**optimization for benchmarking**

If you like to run benchmarks you should set at least the following CMake variables.

.. code-block::

  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-ftree-vectorize -march=native" --build ../alpaka3
  # with CUDA enabled
  # -DCMAKE_CUDA_FLAGS="-ftree-vectorize -march=native"
  # with HIP enabled
  # -DCMAKE_HIP_FLAGS="-ftree-vectorize -march=native"

You can benchmark BabelStream for different number of elements e.g. with a simple loop

.. code-block::

  for((i=1;i<10;++i)) ; do
    ./benchmark/babelstream/babelstream --array-size=$((33554432 * $i)) --number-runs=100
  done
