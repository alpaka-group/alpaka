/* Copyright 2023  Mehmet Yusufoglu, Rene Widera,
 * SPDX-License-Identifier: ISC
 */
/*
 * This example uses cuBLAS library functions in alpaka. A cuBLAS function cublasSgemm is called by using alpaka
 * buffers and queue. Since the code needs only AccGpuCuda backend. Make sure the correct alpaka cmake backend flag is
 * set for alpaka.
 */
#include <alpaka/alpaka.hpp>

#include <cublas_v2.h>

#include <cmath>
#include <iostream>

// Index type
using Idx = std::size_t;
// Set data type
using DataType = float;

// Initialize the matrix in column-major order (1D buffer)
void initializeMatrix(DataType* buffer, Idx rows, Idx cols)
{
    for(Idx j = 0; j < rows; ++j)
    {
        for(Idx i = 0; i < cols; ++i)
        {
            // generate some values and set buffer
            buffer[i + j * cols] = static_cast<DataType>((i + j * cols) % 10);
        }
    }
}

auto main() -> int
{
    using Dim1D = alpaka::DimInt<1>;

    // Define matrix dimensions, A is MxK and B is KxN
    Idx const M = 4; // Rows in A and C
    Idx const N = 2; // Columns in B and C
    Idx const K = 3; // Columns in A and rows in B

    // Define the accelerator and queue
    // Use Cuda Accelerator. Cmake Acc flags should be set to Cuda-Only
    using Acc = alpaka::TagToAcc<alpaka::TagGpuCudaRt, Dim1D, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    // Allocate 1D host memory
    auto bufHostA = alpaka::allocBuf<DataType, Idx>(devHost, M * K);
    auto bufHostB = alpaka::allocBuf<DataType, Idx>(devHost, K * N);
    auto bufHostC = alpaka::allocBuf<DataType, Idx>(devHost, M * N);

    DataType* hostA = std::data(bufHostA);
    DataType* hostB = std::data(bufHostB);
    DataType* hostC = std::data(bufHostC);

    // Initialize host matrices with some values
    initializeMatrix(hostA, M, K);
    initializeMatrix(hostB, K, N);
    std::fill(hostC, hostC + (M * N), 0); // Initialize C with 0s

    // Print initialized matrices
    std::cout << "Matrix A (Host):" << std::endl;
    for(Idx j = 0; j < M; ++j)
    {
        for(Idx i = 0; i < K; ++i)
        {
            std::cout << hostA[i + j * K] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B (Host):" << std::endl;
    for(Idx j = 0; j < K; ++j)
    {
        for(Idx i = 0; i < N; ++i)
        {
            std::cout << hostB[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    // Allocate 1D device memory
    auto bufDevA = alpaka::allocBuf<DataType, Idx>(devAcc, M * K);
    auto bufDevB = alpaka::allocBuf<DataType, Idx>(devAcc, K * N);
    auto bufDevC = alpaka::allocBuf<DataType, Idx>(devAcc, M * N);

    // Copy data to device
    alpaka::memcpy(queue, bufDevA, bufHostA);
    alpaka::memcpy(queue, bufDevB, bufHostB);
    alpaka::memcpy(queue, bufDevC, bufHostC);
    alpaka::wait(queue);

    std::cout << "Copied matrices A and B to the device." << std::endl;

    // Get the native CUDA stream from Alpaka queue
    auto alpakaStream = alpaka::getNativeHandle(queue);

    // cuBLAS setup
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetStream(cublasHandle, alpakaStream);

    // Perform matrix multiplication C = A * B.
    // Set alpha = 1.0f and beta = 0.0f so that the equation C = alpha * A * B + beta * C simplifies to C = A * B.
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N, // No transpose for A and B
        M,
        N,
        K, // Dimensions: C = A * B
        &alpha,
        std::data(bufDevA),
        M, // Leading dimension of A
        std::data(bufDevB),
        K, // Leading dimension of B
        &beta,
        std::data(bufDevC),
        M // Leading dimension of C
    );

    alpaka::wait(queue); // Wait for multiplication to complete
    std::cout << "Matrix multiplication completed." << std::endl;

    // Copy result back to host
    alpaka::memcpy(queue, bufHostC, bufDevC);
    alpaka::wait(queue);
    std::cout << "Copied result matrix C back to the host." << std::endl;

    // Print result matrix C
    std::cout << "Matrix C (Host):" << std::endl;
    for(Idx j = 0; j < M; ++j)
    {
        for(Idx i = 0; i < N; ++i)
        {
            std::cout << hostC[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    // Expected values of elements of C
    std::vector<DataType> expectedResult{20, 23, 6, 9, 56, 68, 30, 42};

    // Verify the result
    bool success = true;
    for(Idx j = 0; j < M; ++j)
    {
        for(Idx i = 0; i < N; ++i)
        {
            if(std::fabs(hostC[i + j * N] - expectedResult[i + j * N]) > 1e-5f)
            { // Allow small floating-point errors
                std::cout << "Mismatch at (" << i << ", " << j << "): " << hostC[i + j * N]
                          << " != " << expectedResult[i + j * N] << std::endl;
                success = false;
            }
        }
    }

    std::cout << "Multiplication of matrices of size " << M << "x" << K << " and " << K << "x" << N
              << (success ? " succeeded!" : " failed!") << std::endl;

    if(!success)
    {
        return EXIT_FAILURE;
    }

    // Cleanup cuBLAS
    cublasDestroy(cublasHandle);
    return EXIT_SUCCESS;
}
