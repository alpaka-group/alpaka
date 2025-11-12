#!/usr/bin/env bash
set -e
# rm -rf build

# mkdir -p build
 cd build/

#source scl_source enable gcc-toolset-14

# CUDA
  # -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  # -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc  \
  # -DCMAKE_CUDA_ARCHITECTURES=89 \

# HIP
  # -DCMAKE_HIP_STANDARD='$ALPAKA_CXX_STANDARD' \
  # -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON \
  # -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/llvm/bin/amdclang++ \
  # -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
  # -Drocrand_DIR=/opt/rocm-6.4.0/lib/cmake/rocrand \
  # -DAMDDeviceLibs_DIR=/opt/rocm-6.4.0/lib/cmake/AMDDeviceLibs \

# maybe not be needed
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
#   -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/llvm/bin/amdclang++ \

# CPU serial
# cmake ~/cern/my-alpaka \
#   -Dalpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB=? \
#   -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
#   -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
#   -Dalpaka_BUILD_EXAMPLES=ON \
#   -Dalpaka_BUILD_BENCHMARKS=ON \
#   -DBUILD_TESTING=ON \
#   -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
#   -DALPAKA_CXX_STANDARD=20 \
#   -L

# CUDA
cmake ~/cern/my-alpaka \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_BUILD_EXAMPLES=ON \
  -Dalpaka_BUILD_BENCHMARKS=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DALPAKA_CXX_STANDARD=20 \
  -L

# HIP 
# cmake ~/cern/my-alpaka \
#   -Dalpaka_ACC_GPU_HIP_ENABLE=ON \
#   -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
#   -Dalpaka_BUILD_EXAMPLES=ON \
#   -Dalpaka_BUILD_BENCHMARKS=ON \
#   -DBUILD_TESTING=ON \
#   -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
#   -Drocrand_DIR=/opt/rocm-6.4.0/lib/cmake/rocrand \
#   -DAMDDeviceLibs_DIR=/opt/rocm-6.4.0/lib/cmake/AMDDeviceLibs \
#   -DCMAKE_HIP_ARCHITECTURES=gfx900 \
#   -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
#   -DALPAKA_CXX_STANDARD=20 \
#   -L

# SYCL
# cmake ~/cern/my-alpaka \
#   -Dalpaka_RELOCATABLE_DEVICE_CODE=?
#   -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
#   -Dalpaka_BUILD_EXAMPLES=ON \
#   -Dalpaka_BUILD_BENCHMARKS=ON \
#   -DBUILD_TESTING=ON \
#   -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
#   -DALPAKA_CXX_STANDARD=20 \
#   -L


#to see what tests exist
#./test/unit/mem/buf/memBufTest --list-tests

# Build only the memBufTest target using all available cores
# make memBufTest -j"$(nproc)"

# Run the built test executable
# ./test/unit/mem/buf/memBufTest

# Builds only the target named 'memBufTest' (your specific unit test)
cmake --build . --target memBufTest -j$(nproc)

# Runs only the test whose name matches 'memBufTest' using CTest
ctest -R memBufTest --output-on-failure

# ./test/unit/mem/buf/memBufTest -s -r compact
# # -s / --success  : show successful assertions & messages
# # -r compact      : compact reporter (use -r console for verbose)

# Alternative way to build the target using Make directly
# make memBufTest

# Builds *all* targets in the project using all available CPU cores
# make -j`nproc`

# Runs all registered tests in the current build directory
# make test