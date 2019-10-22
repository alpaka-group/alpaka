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

## Examples compilation status

|target|location|compiler|compile status|run status
|---|---|---|---|---|
|vectorAdd|example/vectorAdd|
|||GGC 9.1 | compiles| segfault |
|||AOMP|segfault||
|||LLVM 8.0|compiles| segfault loading shared libs before main()|
|||LLVM 9.0|segfault (same as AOMP)||
