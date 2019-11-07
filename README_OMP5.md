# Building configuring for the OMP5 backend

To make the build system enable the OpenMP5 backend (internally called
`CPU_BT_OMP4` for now), one has to tell CMake explicitly about the OpenMP
version supported by the compiler. CMake does not determine it automatically.
```
cmake -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_CPU_BT_OMP4_ENABLE=on \
```
Add flags to set the required compiler and linker flags, e.g:
```
  -DCMAKE_CXX_FLAGS="-fopenmp -fopenmp=libomp -fopenmp-targets=x86_64-pc-linux-gnu"
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
```

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
||AOMP 0.7-4|ok|x86|omp_target_alloc() returns 0|
||AOMP 0.7-4|linker: multiple def. of gpuHeap (1)|amdhsa|--|
||LLVM 9.0|omp tuple warning| x86|segfault loading shared libs before main()|

#### errors:
1. error: Linking globals named 'gpuHeap': symbol multiply defined!
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    clang-9: error: amdgcn-link command failed with exit code 1 (use -v to see 
    invocation)
    clang-9: error: linker command failed with exit code 1 (use -v to see 
    invocation)
