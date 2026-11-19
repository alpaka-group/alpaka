# SIMD RGB Grayscale Example

Packed RGBA to grayscale conversion using SIMD for performance comparison.

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
make reduceRGBSIMD
```

## Run
`./reduceRGBSIMD numElements=1048576`

Demonstrates packed pixel processing with scalar vs SIMD timing.
