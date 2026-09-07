# SIMD Vector Addition Example

Simple element-wise vector addition (c = a + b) using SIMD operations.

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
make vectorAddSIMD
```

## Run
`./vectorAddSIMD 1024`

Shows both explicit SIMD kernel and `forEach`-based approaches.
