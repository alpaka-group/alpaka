#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>

//! IMPORTANT: SIMD Performance Optimization Requirements
//! For optimal SIMD performance, compile with -O3 optimization and -march=native. Without these flags, SIMD operations
//! may fall back to scalar execution, resulting in poor performance. For cross-platform builds or specific targeting,
//! use explicit flags like -mavx2, but -march=native automatically enables all supported instructions on the target
//! CPU.


// SIMD Kernel: uses ForEach function of simdAlgo to iterate over full SIMD packs
// plus an auto-handled tail. If the last pack is not full, it is processed with a scalar(non-simd) loop automatically.
// In the kernel each visited pack is squared element-wise and accumulated into a thread-local sum.
template<typename T>
class SumOfSquaresSIMDKernel
{
public:
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, T const* in, T* result, size_t n) const
    {
        using SimdT = alpaka::simd::PortableSimd<T, Acc>;
        T local = 0;
        alpaka::simd::SimdAlgo<Acc, T> algo(acc, n);

        // Process array in SIMD chunks with automatic tail handling
        // base:  Starting index in the array for this SIMD chunk (e.g., 0, simdWidth, 2*simdWidth, ...)
        // valid: Number of valid elements in this chunk (usually simdWidth, but may be less for the final partial
        // chunk)
        // This handles arrays whose size even is not a multiple of simdWidth, hides tail handling.
        algo.forEach(
            [&](std::size_t base, std::size_t valid)
            {
                SimdT v;
                alpaka::simd::loadSimd<T, Acc>(in + base, valid, v);
                auto squared = v * v;
                // Extract each element from the SIMD vector and accumulate
                // lane = element index within the SIMD chunk (0, 1, 2, ..., valid-1)
                for(std::size_t lane = 0; lane < valid; ++lane)
                    local += squared[lane]; // squared[lane] = element at position 'lane' in the SIMD chunk
            });
        alpaka::atomicAdd(acc, result, local, alpaka::hierarchy::Blocks{});
    }
};

// Non-SIMD Kernel - Pure scalar no vector is defined
template<typename T>
class SumOfSquaresNonSIMDKernel
{
public:
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, T const* in, T* result, size_t dataSize, std::size_t simdWidth) const
    {
        auto threadId = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto numThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        T local = 0;
        // Grid strided loop. If numThreads is less than dataSize/simdWidth, threads processes multiple packs
        for(std::size_t pack = threadId; pack * simdWidth + simdWidth <= dataSize; pack += numThreads)
        {
            std::size_t base = pack * simdWidth;
            for(std::size_t i = 0; i < simdWidth; ++i)
            {
                T x = in[base + i];
                local += x * x;
            }
        }
        // Handle tail elements if any
        if(threadId == 0)
        {
            std::size_t full = (dataSize / simdWidth) * simdWidth;
            for(std::size_t i = full; i < dataSize; ++i)
            {
                T x = in[i];
                local += x * x;
            }
        }
        alpaka::atomicAdd(acc, result, local, alpaka::hierarchy::Blocks{});
    }
};

// Template function to test with different data types
template<typename T, typename Acc, typename Queue, typename DevAcc, typename DevHost>
void testDataType(
    Queue& queue,
    DevAcc const& devAcc,
    DevHost const& devHost,
    std::string const& typeName,
    size_t numElements)
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = size_t;

    Idx const elementsPerThread(1);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    // Allocate buffers
    using BufHost = alpaka::Buf<DevHost, T, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<T, Idx>(devHost, extent));

    // Initialize data
    std::random_device rd;
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<T> dist(1.0, 42.0);
    // Accumulate in higher precision to reduce rounding error
    long double referenceSumAcc = 0.0L;
    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        referenceSumAcc += static_cast<long double>(bufHostA[i]) * static_cast<long double>(bufHostA[i]);
    }
    T referenceSum = static_cast<T>(referenceSumAcc);

    using BufAcc = alpaka::Buf<DevAcc, T, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<T, Idx>(devAcc, extent));

    // Allocate result buffer on device (single element)
    using BufResultAcc = alpaka::Buf<DevAcc, T, Dim, Idx>;
    using BufResultHost = alpaka::Buf<DevHost, T, Dim, Idx>;
    BufResultAcc bufResultAcc(alpaka::allocBuf<T, Idx>(devAcc, alpaka::Vec<Dim, Idx>{1}));
    BufResultHost bufResultHost(alpaka::allocBuf<T, Idx>(devHost, alpaka::Vec<Dim, Idx>{1}));
    T* resultPtr = alpaka::getPtrNative(bufResultAcc);

    alpaka::memcpy(queue, bufAccA, bufHostA);

    const std::size_t simdWidth = alpaka::simd::simdWidth<T, Acc>();
    size_t const blocksForLaunch = (numElements + simdWidth - 1) / simdWidth;

    std::cout << "\n=== Testing with " << typeName << " ===" << std::endl;
    std::cout << "simdWidth for type " << typeName << " is " << simdWidth << std::endl;
    std::cout << "numElements: " << numElements << " (processed in " << blocksForLaunch << " chunks of up to "
              << simdWidth << ")" << std::endl;

    // Variables to store timing for comparison
    double nonSimdTime = 0.0;
    double simdTime = 0.0;

    // Measure Non-SIMD Kernel
    {
        // Initialize result to zero on device
        T zero = 0.0;
        bufResultHost[0] = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresNonSIMDKernel<T> nonSimdKernel;

        // Proper work division: CPU vs GPU
        alpaka::Vec<Dim, Idx> nonSimdExtent;
        if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt, alpaka::TagGpuHipRt>)
        {
            // GPU: Use reasonable thread count similar to SIMD kernel (32K threads max)
            // This avoids massive oversubscription that causes poor performance
            // Grid-stride loop handles remaining work efficiently
            nonSimdExtent = alpaka::Vec<Dim, Idx>(std::min(static_cast<Idx>(32768), numElements));
        }
        else
        {
            // CPU: Check if this is a single-threaded or multi-threaded accelerator
            if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagCpuSerial>)
            {
                // Single-threaded CPU: Use minimal threads
                nonSimdExtent = alpaka::Vec<Dim, Idx>(1);
            }
            else
            {
                // Multi-threaded CPU: Optimized for SIMD performance
                Idx maxThreads = std::thread::hardware_concurrency();
                Idx blockSize = 2 * maxThreads;
                Idx numBlocks = numElements / (simdWidth * blockSize);
                if(numBlocks == 0)
                    numBlocks = 1; // Ensure at least 1 block
                nonSimdExtent = alpaka::Vec<Dim, Idx>(numBlocks * blockSize);
            }
        }

        alpaka::KernelCfg<Acc> const kernelCfg = {nonSimdExtent, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(
            kernelCfg,
            devAcc,
            nonSimdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements,
            simdWidth); // pass original size + width

        std::cout << " " << std::endl;
        std::cout << "Non-SIMD Kernel threads: " << nonSimdExtent[0] << " (each handles up to " << simdWidth
                  << " elements)" << std::endl;

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            nonSimdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements,
            simdWidth); // total size + width

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        // Copy result back to host
        alpaka::memcpy(queue, bufResultHost, bufResultAcc);
        alpaka::wait(queue);
        T result = *alpaka::getPtrNative(bufResultHost);

        std::cout << " " << std::endl;
        std::cout << "Non-SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count()
                  << "s\n";
        long double relErr = std::abs((static_cast<long double>(result) - referenceSumAcc) / referenceSumAcc);
        std::cout << "Non-SIMD Kernel Result: " << result << " (relErr vs refAcc " << relErr << ")\n";

        // Store non-SIMD time for comparison
        nonSimdTime = std::chrono::duration<double>(endT - beginT).count();
    }

    // Measure SIMD Kernel - Now using REDUCED thread count for true SIMD benefit
    {
        // Initialize result to zero on device
        T zero = 0.0;
        *alpaka::getPtrNative(bufResultHost) = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresSIMDKernel<T> simdKernel;

        // IMPORTANT: Use backend-aware work division for optimal performance
        // GPU backends: Use high thread count for better occupancy
        // CPU backends: Use SIMD-friendly reduced thread count
        constexpr bool isSerialAcc = alpaka::simd::isCpuSerialAcc<Acc>();
        constexpr bool isGpuAcc = alpaka::simd::isGpuAcc<Acc>();

        alpaka::Vec<Dim, Idx> simdExtent;
        if constexpr(isSerialAcc)
        {
            simdExtent = alpaka::Vec<Dim, Idx>(Idx(1));
        }
        else if constexpr(isGpuAcc)
        {
            // GPU: Use high thread count for better occupancy (up to 32K threads)
            simdExtent = alpaka::Vec<Dim, Idx>(std::min(static_cast<Idx>(32768), numElements));
        }
        else
        {
            // CPU: Use SIMD-friendly reduced thread count
            simdExtent = alpaka::Vec<Dim, Idx>(blocksForLaunch);
        }

        alpaka::KernelCfg<Acc> const simdKernelCfg = {simdExtent, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(
            simdKernelCfg,
            devAcc,
            simdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements);

        std::cout << " " << std::endl;
        std::cout << "SIMD Kernel threads: " << simdExtent[0] << " (tail auto-handled)" << std::endl;
        std::cout << "Each SIMD thread processes up to " << simdWidth << " elements" << std::endl;

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            simdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        // Copy result back to host
        alpaka::memcpy(queue, bufResultHost, bufResultAcc);
        alpaka::wait(queue);
        T result = *alpaka::getPtrNative(bufResultHost);

        std::cout << "SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count() << "s\n";
        long double relErr = std::abs((static_cast<long double>(result) - referenceSumAcc) / referenceSumAcc);
        std::cout << "SIMD Kernel Result: " << result << " (relErr vs refAcc " << relErr << ")\n";

        // Store SIMD time for comparison
        simdTime = std::chrono::duration<double>(endT - beginT).count();
    }

    std::cout << " " << std::endl;
    std::cout << "Reference Sum of Squares: " << referenceSum << "\n";

    // Print SIMD improvement ratio
    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    double improvementRatio = nonSimdTime / simdTime;
    std::cout << "non_simd_time_s = " << nonSimdTime << '\n';
    std::cout << "simd_time_s     = " << simdTime << '\n';
    std::cout << "speedup         = " << improvementRatio << " (non_simd/simd)" << '\n';
    std::cout << "simd policy     = " << alpaka::simd::policyName<double, Acc>() << '\n';
    if(improvementRatio < 1.5)
        std::cout << "note            = consider -O3 -march=native for higher speedup" << '\n';
    std::cout << "---------------------------" << std::endl;
}

// Example function to compare SIMD and non-SIMD kernels
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, size_t numElements) -> int
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = size_t; // Use size_t consistently

    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    if constexpr(!alpaka::simd::isCpuSerialAcc<Acc>() && std::is_same_v<TAccTag, alpaka::TagCpuSerial>)
        return 0; // skip silently if serial tag but backend not built
    using DevAcc = alpaka::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Print SIMD system information
    alpaka::simd::printSimdSystemInfo();

    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);
    QueueAcc queue(devAcc);

    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Test with double
    testDataType<double, Acc, QueueAcc, DevAcc, DevHost>(queue, devAcc, devHost, "double", numElements);

    // Test with float
    testDataType<float, Acc, QueueAcc, DevAcc, DevHost>(queue, devAcc, devHost, "float", numElements);

    return EXIT_SUCCESS;
}

auto main(int argc, char* argv[]) -> int
{
#ifdef ALPAKA_CI
    size_t numElements = 1 << 20; // 1M elements for CI (2^20)
#else
    size_t numElements = 1 << 22; // 4M elements for local (2^22)
#endif

    // Parse command-line argument
    if(argc > 1)
    {
        std::string arg = argv[1];
        if(arg.find("numElements=") == 0)
        {
            try
            {
                numElements = std::stoul(arg.substr(12));
            }
            catch(std::invalid_argument const&)
            {
                std::cerr << "Invalid number of elements: " << arg.substr(12) << std::endl;
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const&)
            {
                std::cerr << "Number of elements out of range: " << arg.substr(12) << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << " numElements=<value>" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();

    // Print numElements; allow arbitrary (tail handled inside kernels)
    std::cout << "numElements: " << numElements;
    if(numElements == 0)
    {
        std::cerr << "\nError: numElements must be > 0." << std::endl;
        return EXIT_FAILURE;
    }
    if((numElements & (numElements - 1)) != 0)
    {
        std::cout << " (not a power of two)";
    }
    std::cout << std::endl;

    // Iterate all enabled accelerator tags; skip those whose underlying accelerator is incomplete (e.g., TagCpuSerial
    // when disabled)
    return alpaka::executeForEachAccTag([&](auto const& tag) { return example(tag, numElements); });
}
