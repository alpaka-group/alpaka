Cross Compile
=============

Sometimes it is necessary to cross compile alpaka for a different architecture.
Here we provide a small example how to compile on x86 for RISC-V on the test system https://riscv.epcc.ed.ac.uk/

.. code-block:: bash

  module load riscv64-linux/gnu-12.2
  # download the latest CMake 3.3X and set it to your environment PATH variable
  export RISCV_GNU_INSTALL_ROOT=/usr/local/share/riscv-compiler/gnu-12.2/build/rv64gc-linux/
  # Some tests failing to compile, not because of C++ the compile or linker runs into parsing issues.
  # Therefore tests are disabled by default.
  cmake -Dalpaka_BENCHMARKS=ON  -Dalpaka_EXAMPLES=ON ../alpaka -DCMAKE_CXX_FLAGS="-march=rv64gc -mabi=lp64d -fopenmp" -DCMAKE_EXE_LINKER_FLAGS="-lpthread -fopenmp" -DCMAKE_TOOLCHAIN_FILE=../alpaka/cmake/riscv64-gnu.toolchain.cmake  -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -DCMAKE_BUILD_TYPE=Release
  make -j
  # one of the Milk-V Pionee nodes
  srun -n 1 --nodelist=rvc24 --time=00:30:00 --cpu-bind=none --cpus-per-task=64 -q riscv --pty bash
  export OMP_NUM_THREADS=64
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  ctest --output-on-failure
