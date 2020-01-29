# Building configuring for the OMP5 backend

To make the build system enable the OpenMP5 backend (internally called
`CPU_BT_OMP4` for now), one has to tell CMake explicitly about the OpenMP
version supported by the compiler. CMake does not determine it automatically.
```
cmake -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_CPU_BT_OMP4_ENABLE=on \
	-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=on \
	-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=off \
	-DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=off \
	-DALPAKA_ACC_GPU_CUDA_ENABLE=off \
	-DALPAKA_ACC_GPU_HIP_ENABLE=off \
```
All other backends are disable for faster compilation/testing and reduced
environment requirements. Add flags to set the required compiler and linker flags, e.g:
- clang/AOMP, target x86:
  ```
    -DCMAKE_CXX_FLAGS="-fopenmp -fopenmp=libomp -fopenmp-targets=x86_64-pc-linux-gnu" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
  ```
- clang/AOMP, target ppc64le:
  ```
    -DCMAKE_CXX_FLAGS="-fopenmp -fopenmp=libomp -fopenmp-targets=ppc64le-pc-linux-gnu" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
  ```
- clang, target nvptx:
  ```
    -DCMAKE_CXX_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O2" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
  ```
- AOMP, target amdhsa:
  ```
    -DCMAKE_CXX_FLAGS="-fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 --save-temps" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
  ```
- GCC, target nvptx:
  ```
    -DCMAKE_CXX_FLAGS="-foffload=nvptx-none -foffload=-lm -fno-lto"
  ```
- XL, offload:
  ```
    -DCMAKE_CXX_FLAGS="-qoffload -qsmp"
  ```
- XL, no offload:
  ```
    -DCMAKE_CXX_FLAGS=""
  ```

With clang 9, AOMP 0.7-4, use libc++ instead of libstdc++, the latter will make
the compiler crash: https://bugs.llvm.org/show_bug.cgi?id=43771 .

## 1. Test target

```
make vectorAdd
./example/vectorAdd/vectorAdd
```
If the run is successful, the last output line will be `Execution results
correct!` otherwise it will print items where the result from the offload code
disagrees with the expected result and print `Execution results
incorrect!` at the end.

## 2. Examples compilation status

### branch omp4

|target|compiler|compile status|target|run status|
|---|---|---|---|---|
|vectorAdd|
||GGC 9.2 (RWTH Aachen)| Vec<> not mappable type|nvptx|--|
||AOMP 0.7-4|ok|x86|omp_target_alloc() returns 0|
||AOMP 0.7-4|linker: multiple def. of gpuHeap (1)|amdhsa|--|
||AOMP 0.7-5|ok|x86|ok|
||AOMP 0.7-5|ok	|amdhsa|ok|
||LLVM 9.0|omp tuple warning| x86|segfault loading shared libs before main()|
||LLVM 9.0 (RWTH Aachen)|omp tuple warning| x86|ok|
||LLVM 9.0 (RWTH Aachen)|ok (with -O2, else 2)| nvptx|ok|
||XL 16.1.1-5 (Summit)|ok| nvptx|ok (num_threads workaround) (3)|
||XL 16.1.1-5 (Summit)|ok| ppc64le| sigsegv (device mem alloc'son GPU)|

#### errors:
1. error: Linking globals named 'gpuHeap': symbol multiply defined!
   ```
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    clang-9: error: amdgcn-link command failed with exit code 1 (use -v to see 
    invocation)
    clang-9: error: linker command failed with exit code 1 (use -v to see 
    invocation)
   ```
2. ptxas:
   ```
    ptxas main-d4ddfc.s, line 6798; warning : Instruction 'vote' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
    [100%] Linking CXX executable vectorAdd
    nvlink error   : Undefined reference to '__assert_fail' in 'main-83ce50.cubin'
    clang-9: error: nvlink command failed with exit code 255 (use -v to see invocation)
   ```
3. IBM XL: When setting num_threads, either in #pragma omp parallel or via
   omp_set_num_threads to any value the runtime only executes one thread per
   team. Workaround is to not do that with XL, which leads to $OMP_NUM_THREADS
   being run per team. Minimal example:
   https://github.com/jkelling/omp5tests/blob/master/parallel/parallel.cpp
4. g++:
   ```
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp: In member function 'void alpaka::kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs>::operator()() const [with TDim = std::integral_constant<long unsigned int, 1>; TIdx = long unsigned int; TKernelFnObj = VectorAddKernel; TArgs = {unsigned int*, unsigned int*, unsigned int*, const long unsigned int&}]':
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp:157:80: error: 'threadElemExtent' referenced in target region does not have a mappable type
       157 |                         printf("threadElemCount_dev %d\n", int(threadElemExtent[0u]));
           |                                                                ~~~~~~~~~~~~~~~~^
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp:161:57: error: 'blockThreadExtent' referenced in target region does not have a mappable type
       161 |                             acc::AccCpuOmp4<TDim, TIdx> acc(
           |                                                         ^~~
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp:161:57: error: 'gridBlockExtent' referenced in target region does not have a mappable type
   ```

## 3. Integration and Unit Tests

Run `make` and upon success `ctest`.

|test|compiler|compile status|target|run status|
|---|---|---|---|---|
|ALL|
||LLVM 9.0 (RWTH Aachen)|ok|x86|pass|
||LLVM 9.0 (RWTH Aachen)|ok (with -O2, else 2)| nvptx|omp_target_alloc() fails [1]|
||LLVM 9.0.0-1 (Summit)|ok|ppc64le|fail [3]|
||LLVM 9.0.0-1 (Summit)|fail [4]|nvptx|--|
||LLVM 9.0.0-2 (Summit)|ok|ppc64le|fail [3]|
||LLVM 9.0.0-2 (Summit)|fail [4]|nvptx|--|
||AOMP 0.7-5|linker error with static lib (7)x86|--|
||AOMP 0.7-5|linker error with static lib (8)|amdhsa|--|
||GCC 9.1.0 (Summit)|fail [5]|nvptx|--|
||XL 16.1.1-5 (Summit)|no-halt [6]|nvptx|--|
||XL 16.1.1-5 (Summit)|no-halt [6]|ppc64le|--|

#### errors:
1. omp_target_alloc() always returns NULL with nvptx backend.
2. ptxas:
   ```
    ptxas main-d4ddfc.s, line 6798; warning : Instruction 'vote' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
    [100%] Linking CXX executable vectorAdd
    nvlink error   : Undefined reference to '__assert_fail' in 'main-83ce50.cubin'
    clang-9: error: nvlink command failed with exit code 255 (use -v to see invocation)
   ```
3. Libomptarget fatal error 1: failure of target construct while offloading is mandatory
   * Triggered by muteable member in acc/AccCpuOmp4.hpp:133, instanciation at kernel/TaskKernelCpuOmp4.hpp:282 .
   * Triggered by call to meta::apply(...) at kernel/TaskKernelCpuOmp4.hpp:335, even with empty lambda, no clojure, tuple<int,int> and no parameter pack.
   * Fix: compile with --std=c++14 instead of 11
4. nvlink error   : Undefined reference to '_ZNSt17integral_constantIbLb0EE5valueE' in '/tmp/vectorAdd-016d41.cubin'
   * this maybe because of libstd++, need LLVM with libc++
5. g++:
   ```
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp: In member function 'void alpaka::kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs>::operator()() const [with TDim = std::integral_constant<long unsigned int, 1>; TIdx = long unsigned int; TKernelFnObj = VectorAddKernel; TArgs = {unsigned int*, unsigned int*, unsigned int*, const long unsigned int&}]':
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp:157:80: error: 'threadElemExtent' referenced in target region does not have a mappable type
       157 |                         printf("threadElemCount_dev %d\n", int(threadElemExtent[0u]));
           |                                                                ~~~~~~~~~~~~~~~~^
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp:161:57: error: 'blockThreadExtent' referenced in target region does not have a mappable type
       161 |                             acc::AccCpuOmp4<TDim, TIdx> acc(
           |                                                         ^~~
      include/alpaka/kernel/TaskKernelCpuOmp4.hpp:161:57: error: 'gridBlockExtent' referenced in target region does not have a mappable type
   ```
   * Type Vec<...> should be mappable.
6. XL does not appear to terminate when compiling targets like `blockShared` in
   which tests are executed through the fixture in
   ~alpaka/test/common/include/alpaka/test/KernelExecutionFixture.hpp .
   Removing the call
   alpaka/test/unit/block/shared/src/BlockSharedMemDyn.cpp:92-94 yields finite
   compilation time for BlockSharedMemDyn.cpp.o . XL is extremely slow
   compiling code using the test framework catch2 used in Alpaka.
7. aomp 0.7-5 x86:
   ```
   /usr/bin/ld: cannot find libcommon-openmp-x86_64-pc-linux-gnu-sm_20.o: No such file or directory
   /usr/bin/ld: cannot find libcommon-host-x86_64-unknown-linux-gnu.o: No such file or directory
   clang-9: error: linker command failed with exit code 1 (use -v to see invocation)
   clang-9: error: linker command failed with exit code 1 (use -v to see invocation)
   test/integ/matMul/CMakeFiles/matMul.dir/build.make:85: recipe for target 'test/integ/matMul/matMul' failed
   ```
8. aomp 0.7-5 HSA:
   ```
   /home/kelling/rocm/aomp_0.7-5/bin/clang-build-select-link: libcommon-openmp-amdgcn-amd-amdhsa-gfx900.o:1:2: error: expected integer
   !<arch>
    ^
   /home/kelling/rocm/aomp_0.7-5/bin/clang-build-select-link: error:  loading file 'libcommon-openmp-amdgcn-amd-amdhsa-gfx900.o'
   /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
   /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
   clang-9: error: amdgcn-link command failed with exit code 1 (use -v to see invocation)
   clang-9: error: linker command failed with exit code 1 (use -v to see invocation)
   ```
