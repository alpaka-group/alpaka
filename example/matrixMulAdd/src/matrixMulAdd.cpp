/* Copyright 2023  Mehmet Yusufoglu, Rene Widera,
 * SPDX-License-Identifier: ISC
 */
/*
 * This example demonstrates how to use cuBLAS together with alpaka. Two GEMM
 * operations are executed and each result is reduced by an alpaka kernel.
 * The execution times for the GEMMs, reductions and the entire pipeline are
 * printed.
 */
#include <alpaka/alpaka.hpp>

#include <cublas_v2.h>

#include <chrono>
#include <iostream>

using Dim1D = alpaka::DimInt<1>;
using Idx = std::size_t;
using Data = float;

//! Kernel summing four elements of a 2x2 matrix
struct SumKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const & acc, Data const *mat, Data *result) const
    {
        if(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0)
        {
            result[0] = mat[0] + mat[1] + mat[2] + mat[3];
        }
    }
};

int main()
{
    using Acc = alpaka::TagToAcc<alpaka::TagGpuCudaRt, Dim1D, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    // 2x2 matrices stored in column major order
    auto bufHostA = alpaka::allocBuf<Data, Idx>(devHost, 4u);
    auto bufHostB = alpaka::allocBuf<Data, Idx>(devHost, 4u);
    auto bufHostC1 = alpaka::allocBuf<Data, Idx>(devHost, 4u);
    auto bufHostC2 = alpaka::allocBuf<Data, Idx>(devHost, 4u);
    auto bufHostSum1 = alpaka::allocBuf<Data, Idx>(devHost, 1u);
    auto bufHostSum2 = alpaka::allocBuf<Data, Idx>(devHost, 1u);

    Data* hostA = std::data(bufHostA);
    Data* hostB = std::data(bufHostB);
    Data* hostC1 = std::data(bufHostC1);
    Data* hostC2 = std::data(bufHostC2);
    Data* hostSum1 = std::data(bufHostSum1);
    Data* hostSum2 = std::data(bufHostSum2);

    hostA[0] = 1.f; hostA[1] = 3.f; hostA[2] = 2.f; hostA[3] = 4.f;
    hostB[0] = 5.f; hostB[1] = 7.f; hostB[2] = 6.f; hostB[3] = 8.f;
    for(int i = 0; i < 4; ++i) { hostC1[i] = 0.f; hostC2[i] = 0.f; }
    hostSum1[0] = 0.f;
    hostSum2[0] = 0.f;

    auto bufDevA = alpaka::allocBuf<Data, Idx>(devAcc, 4u);
    auto bufDevB = alpaka::allocBuf<Data, Idx>(devAcc, 4u);
    auto bufDevC1 = alpaka::allocBuf<Data, Idx>(devAcc, 4u);
    auto bufDevC2 = alpaka::allocBuf<Data, Idx>(devAcc, 4u);
    auto bufDevSum1 = alpaka::allocBuf<Data, Idx>(devAcc, 1u);
    auto bufDevSum2 = alpaka::allocBuf<Data, Idx>(devAcc, 1u);

    alpaka::memcpy(queue, bufDevA, bufHostA);
    alpaka::memcpy(queue, bufDevB, bufHostB);
    alpaka::memcpy(queue, bufDevC1, bufHostC1);
    alpaka::memcpy(queue, bufDevC2, bufHostC2);
    alpaka::memcpy(queue, bufDevSum1, bufHostSum1);
    alpaka::memcpy(queue, bufDevSum2, bufHostSum2);
    alpaka::wait(queue);

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, alpaka::getNativeHandle(queue));

    SumKernel sumKernel{};
    alpaka::KernelCfg<Acc> cfgAdd{alpaka::Vec<Dim1D, Idx>(1), Idx{1}};
    auto workDivAdd = alpaka::getValidWorkDiv(cfgAdd, devAcc, sumKernel, alpaka::getPtrNative(bufDevC1), alpaka::getPtrNative(bufDevSum1));

    auto startTotal = std::chrono::high_resolution_clock::now();

    // First GEMM: C1 = A * B
    alpaka::wait(queue);
    auto startMul1 = std::chrono::high_resolution_clock::now();
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, alpaka::getPtrNative(bufDevA), 2, alpaka::getPtrNative(bufDevB), 2, &beta, alpaka::getPtrNative(bufDevC1), 2);
    alpaka::wait(queue);
    auto endMul1 = std::chrono::high_resolution_clock::now();

    // Sum elements of C1
    auto startAdd1 = std::chrono::high_resolution_clock::now();
    alpaka::exec<Acc>(queue, workDivAdd, sumKernel, alpaka::getPtrNative(bufDevC1), alpaka::getPtrNative(bufDevSum1));
    alpaka::wait(queue);
    auto endAdd1 = std::chrono::high_resolution_clock::now();

    // Second GEMM: C2 = C1 * A
    auto startMul2 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, alpaka::getPtrNative(bufDevC1), 2, alpaka::getPtrNative(bufDevA), 2, &beta, alpaka::getPtrNative(bufDevC2), 2);
    alpaka::wait(queue);
    auto endMul2 = std::chrono::high_resolution_clock::now();

    // Sum elements of C2
    auto startAdd2 = std::chrono::high_resolution_clock::now();
    workDivAdd = alpaka::getValidWorkDiv(cfgAdd, devAcc, sumKernel, alpaka::getPtrNative(bufDevC2), alpaka::getPtrNative(bufDevSum2));
    alpaka::exec<Acc>(queue, workDivAdd, sumKernel, alpaka::getPtrNative(bufDevC2), alpaka::getPtrNative(bufDevSum2));
    alpaka::wait(queue);
    auto endAdd2 = std::chrono::high_resolution_clock::now();

    alpaka::memcpy(queue, bufHostC1, bufDevC1);
    alpaka::memcpy(queue, bufHostC2, bufDevC2);
    alpaka::memcpy(queue, bufHostSum1, bufDevSum1);
    alpaka::memcpy(queue, bufHostSum2, bufDevSum2);
    alpaka::wait(queue);

    auto endTotal = std::chrono::high_resolution_clock::now();

    std::cout << "Time GEMM1: " << std::chrono::duration<double>(endMul1 - startMul1).count() << "s\n";
    std::cout << "Time ADD1: " << std::chrono::duration<double>(endAdd1 - startAdd1).count() << "s\n";
    std::cout << "Time GEMM2: " << std::chrono::duration<double>(endMul2 - startMul2).count() << "s\n";
    std::cout << "Time ADD2: " << std::chrono::duration<double>(endAdd2 - startAdd2).count() << "s\n";
    std::cout << "Total time: " << std::chrono::duration<double>(endTotal - startTotal).count() << "s\n";

    std::cout << "Result C1:" << std::endl;
    std::cout << hostC1[0] << " " << hostC1[2] << std::endl;
    std::cout << hostC1[1] << " " << hostC1[3] << std::endl;
    std::cout << "Sum1: " << hostSum1[0] << std::endl;

    std::cout << "Result C2:" << std::endl;
    std::cout << hostC2[0] << " " << hostC2[2] << std::endl;
    std::cout << hostC2[1] << " " << hostC2[3] << std::endl;
    std::cout << "Sum2: " << hostSum2[0] << std::endl;

    cublasDestroy(handle);
    return EXIT_SUCCESS;
}
