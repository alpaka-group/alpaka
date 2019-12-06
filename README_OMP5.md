# Building configuring for the OMP5 backend

To make the build system enable the OpenMP5 backend (internally called
`CPU_BT_OMP4` for now), one has to tell CMake explicitly about the OpenMP
version supported by the compiler. CMake does not determine it automatically.
```
cmake -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_CPU_BT_OMP4_ENABLE=on \
	-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=off \
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

With clang 9, AOMP 0.7-4, use libc++ instead of libstdc++, the latter will make
the compiler crash: https://bugs.llvm.org/show_bug.cgi?id=43771 .


## Test target

```
make vectorAdd
./example/vectorAdd/vectorAdd
```
If the run is successful, the last output line will be `Execution results
correct!` otherwise it will print items where the result from the offload code
disagrees with the expected result and print `Execution results
incorrect!` at the end.

## Examples compilation status

### branch omp4

|target|compiler|compile status|target|run status|
|---|---|---|---|---|
|vectorAdd|
||GGC 9.1 | ok|nvptx| GPUptr, but not on GPU: segfault |
||GGC 9.2 (claix)| ok|nvptx| GPUptr, but not on GPU: segfault (GPU context created, nvprof shows no kernels)|
||AOMP 0.7-4|ok|x86|omp_target_alloc() returns 0|
||AOMP 0.7-4|linker: multiple def. of gpuHeap (1)|amdhsa|--|
||LLVM 9.0|omp tuple warning| x86|segfault loading shared libs before main()|
||LLVM 9.0 (claix)|omp tuple warning| x86|ok|
||LLVM 9.0 (claix)|ok (with -O2, else 2)| nvptx|ok|

#### errors:
1. error: Linking globals named 'gpuHeap': symbol multiply defined!
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    clang-9: error: amdgcn-link command failed with exit code 1 (use -v to see 
    invocation)
    clang-9: error: linker command failed with exit code 1 (use -v to see 
    invocation)
2. ptxas /tmp/yn622878/login-g_6260/main-d4ddfc.s, line 6798; warning : Instruction 'vote' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
    [100%] Linking CXX executable vectorAdd
    nvlink error   : Undefined reference to '__assert_fail' in '/tmp/yn622878/login-g_6260/main-83ce50.cubin'
    clang-9: error: nvlink command failed with exit code 255 (use -v to see invocation)

## Integration and Unit Tests

Run `make` and upon success `ctest`.

|test|compiler|compile status|target|run status|
|---|---|---|---|---|
|ALL|
||LLVM 9.0 (claix)|ok|x86|pass|
||LLVM 9.0 (claix)|ok (with -O2, else 2)| nvptx|omp_target_alloc() fails [1]|

#### errors:
1. omp_target_alloc() always returns NULL with nvptx backend.
