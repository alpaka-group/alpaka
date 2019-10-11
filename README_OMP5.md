# Building configuring for the OMP5 backend

To make the build system enable the OpenMP5 backend (internally called
`CPU_BT_OMP4` for now), one has to tell CMake explicitly about the OpenMP
version supported by the compiler. CMake does not determine it automatically.
```
cmake -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_CPU_BT_OMP4_ENABLE=on \
```
Add flags to set the required compiler and linker flags, e.g:
```
  -DCMAKE_CXX_FLAGS="-fopenmp -fopenmp=libomp -fopenmp-targets=x86_64"
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
```

## Examples compilation status

|target|location|phase|GGC 9.0|AOMP|LLVM 8.0|LLVM 9.0| 
|---|---|---|---|---|---|---|
|vectorAdd|example/vectorAdd|testing|compiles,segfault|ICE|??|??|
