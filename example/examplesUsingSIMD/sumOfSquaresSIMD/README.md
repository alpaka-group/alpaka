# SIMD Sum of Squares Example

Demonstrates SIMD-accelerated sum-of-squares computation vs scalar baseline.

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
make sumOfSquaresSIMD
```

## Run
`./sumOfSquaresSIMD numElements=1048576`

Expected speedup: ~1.5x for float, ~1.1-1.6x for double (CPU-dependent).
3. **Scalar fallback** - For non-x86 architectures or CUDA compilation

### Force Fallback Testing

You can test the fallback implementation by disabling std::experimental::simd at configure time:

```bash
cmake -B build -Dalpaka_USE_STD_SIMD=OFF ...
cmake --build build --target sumOfSquaresSIMD
./build/example/examplesUsingSIMD/sumOfSquaresSIMD/sumOfSquaresSIMD numElements=1048576
```

## Supported Accelerators

- AccCpuSerial
- AccCpuThreads  
- AccCpuOmp2Blocks
- AccCpuOmp2Threads
- AccCpuTbbBlocks

CUDA and HIP accelerators use scalar fallback as SIMD is CPU-specific.
