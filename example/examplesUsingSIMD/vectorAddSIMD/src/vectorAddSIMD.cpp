/* Vector Add SIMD Example: Has 2 kernels using alpaka simd implementation. 1: Kernel using explicit SIMD types.
 * 2:Kernel using simd ForEach function. Keeps VectorAddSimdKernel (not launched) and uses
 * VectorAddSimdKernelUsingForEach in main.
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

// SIMD Option-1: Kernel using simd types and ForEach function
// This kernel performs element-wise vector addition (c=a+b) in SIMD packs; ForEach of SimdAlgo applies the lambda to
// each full SIMD pack and transparently handles the tail. It hides pack looping, grid distribution, and scalar cleanup
// so only the per-pack expression (vA+vB) is written here.
struct VectorAddSimdKernelUsingAlgo
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c, std::size_t n) const
    {
        using SimdT = alpaka::simd::PortableSimd<T, TAcc>;
        alpaka::simd::SimdAlgo<TAcc, T> algo(acc, n);

        // Process arrays in SIMD chunks with automatic tail handling
        // base:  Starting index in the array for this SIMD chunk (e.g., 0, 4, 8, 12, ...)
        // valid: Number of valid elements in this chunk (usually simdWidth, but may be less for the final partial
        // chunk)
        //        For example, if array has 10 elements and simdWidth=4, the chunks would be:
        //        - base=0, valid=4  (elements 0-3)
        //        - base=4, valid=4  (elements 4-7)
        //        - base=8, valid=2  (elements 8-9, final partial chunk)
        algo.forEach(
            [&](std::size_t base, std::size_t valid)
            {
                SimdT va, vb;
                alpaka::simd::loadSimd<T, TAcc>(a + base, valid, va);
                alpaka::simd::loadSimd<T, TAcc>(b + base, valid, vb);
                auto sum = va + vb;
                alpaka::simd::storeSimd<T, TAcc>(sum, valid, c + base);
            });
    }
};

// Simd Option-2: SIMD Kernel using simd types and explicit SIMD operations
struct VectorAddSimdKernelUsingExplicitTypes
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c, std::size_t dataSize) const
    {
        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        auto width = alpaka::simd::simdWidth<T, TAcc>();
        auto threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto numThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        for(std::size_t pack = threadIdx; pack * width + width <= dataSize; pack += numThreads)
        {
            auto baseIndex = pack * width;
            SimdType va;
            va.load(a + baseIndex);
            SimdType vb;
            vb.load(b + baseIndex);
            (va + vb).store(c + baseIndex);
        }
        // Handle tail elements if any
        if(threadIdx == 0)
        {
            std::size_t full = (dataSize / width) * width;
            for(std::size_t i = full; i < dataSize; ++i)
                c[i] = a[i] + b[i];
        }
    }
};

template<alpaka::concepts::Tag TAccTag>
int runExample(TAccTag const&, std::size_t numElements)
{
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    if constexpr(!alpaka::simd::isCpuSerialAcc<Acc>() && std::is_same_v<TAccTag, alpaka::TagCpuSerial>)
        return 0; // skip serial tag if backend not compiled
    auto platform = alpaka::Platform<Acc>{};
    auto devAcc = alpaka::getDevByIdx(platform, 0u);
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    Queue queue(devAcc);

    // numElements supplied by caller (allows large perf runs)
    std::vector<float> a(numElements), b(numElements), c(numElements, 0.0f);
    for(std::size_t i = 0; i < numElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto bufA = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    auto bufB = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    auto bufC = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    alpaka::memcpy(queue, bufA, a);
    alpaka::memcpy(queue, bufB, b);

    auto width = alpaka::simd::simdWidth<float, Acc>();
    Idx threads = static_cast<Idx>((numElements + width - 1) / width);
    if(threads == 0)
        threads = 1;
    auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{threads, Idx(1), Idx(1)};

    // Define the kernel to use
    auto kernel = VectorAddSimdKernelUsingAlgo{};

    // PRINT INFORMATION
    std::cout << "==== SIMD System Information ====\n";
    std::cout << "Accelerator: " << alpaka::getAccName<Acc>() << "\n";
    std::cout << "Simd Policy: " << alpaka::simd::policyName<float, Acc>() << "\n";
    std::cout << "simdWidth(float) = " << width << "\n";
    std::cout << "Elements: " << numElements << " (" << threads << " chunks)\n";
    // Automatically extract kernel name using typeid of kernel. Since there are 2 kernel classes
    std::string kernelTypeStr = typeid(decltype(kernel)).name();
    std::string kernelName;

    // Check for known kernel class names in the mangled name
    if(kernelTypeStr.find("VectorAddSimdKernelUsingExplicitTypes") != std::string::npos)
    {
        kernelName = "VectorAddSimdKernelUsingExplicitTypes";
    }
    else if(kernelTypeStr.find("VectorAddSimdKernelUsingAlgo") != std::string::npos)
    {
        kernelName = "VectorAddSimdKernelUsingAlgo";
    }
    std::cout << "Launching kernel: " << kernelName << "\n";
    std::cout << "==================================\n";


    auto const t0 = std::chrono::high_resolution_clock::now();
    alpaka::exec<Acc>(
        queue,
        workDiv,
        kernel,
        alpaka::getPtrNative(bufA),
        alpaka::getPtrNative(bufB),
        alpaka::getPtrNative(bufC),
        numElements);
    alpaka::wait(queue);
    auto const t1 = std::chrono::high_resolution_clock::now();
    alpaka::memcpy(queue, c, bufC);
    alpaka::wait(queue);

    for(std::size_t i = 0; i < numElements; ++i)
    {
        float expected = a[i] + b[i];
        if(std::abs(c[i] - expected) > 1e-5f)
        {
            std::cout << "Mismatch at " << i << "\n";
            return 1;
        }
    }
    auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "VectorAdd result OK in " << dt << " ms\n";
    return 0;
}

int main(int argc, char** argv)
{
    std::size_t numElements = 1024;
    if(argc > 1)
    {
        try
        {
            numElements = std::stoull(argv[1]);
        }
        catch(...)
        {
            std::cerr << "Usage: " << argv[0] << " [numElements]" << std::endl;
            return 1;
        }
    }
    std::cout << "numElements=" << numElements << "\n";
    return alpaka::executeForEachAccTag([&](auto const& tag) { return runExample(tag, numElements); });
}
