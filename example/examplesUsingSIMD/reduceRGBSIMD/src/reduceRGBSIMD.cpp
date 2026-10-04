// -------------------------------------------------------------------
// reduceRGBSIMD Example
//
// Demonstrates a portable packed 32-bit RGBA grayscale conversion using alpaka::simd::PortableSimd.
// The same source supports:
//   * std::experimental::simd policy (pure CPU builds, not mixed with CUDA)
//   * Array fallback backend (used when alpaka_USE_STD_SIMD=OFF or in mixed CUDA builds)
//   * GPU/CPU accelerator tag iteration via executeForEachAccTag
//
// Design notes:
//   - Uses a grid-stride SIMD scheduler (each thread may handle multiple SIMD packs). We size the
//     total thread count close to (or capped below) the number of SIMD packs to avoid excessive
//     per-thread looping while keeping launch overhead low for huge inputs
//   - Packed pixel format: 0xAARRGGBB; grayscale keeps alpha (AA G G G G G G)
//   - Integer weights chosen for fast integer math avoiding floating point in hot loop.
//   - Fallback SIMD width is heuristically derived from available ISA (SSE2/AVX2/AVX512) but implemented
//     purely with std::array loops – no intrinsics – preserving portability while giving loop unrolling hints.
//   - In mixed CUDA builds NVCC disables std::experimental::simd; the enhanced fallback is used automatically.
//   - Alignment is reported for diagnostic purposes; fallback loads are scalar-friendly so correctness
//     does not depend on alignment, but aligned buffers can help real vector backends.
//   - IMPORTANT BUILD FLAG REMINDER:
//       * Always compile with -O3 and -march=native for meaningful CPU SIMD speedups.
//       * In mixed CUDA/HIP + CPU builds you must forward the march flag, e.g.:
//           CUDA: -DCMAKE_CUDA_FLAGS="-Xcompiler -march=native"
//           HIP : -DCMAKE_HIP_FLAGS="-Xclang -march=native" (or toolchain equivalent)
//         Setting only CMAKE_CXX_FLAGS is insufficient because the device compiler wraps host compilation.
// -------------------------------------------------------------------------------------------------

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <type_traits>

//! Optimal work division for 3 accelerator types: GPU, CPU multi-threaded, CPU serial
template<typename Acc>
auto getOptimalWorkDivForRGBA(alpaka::Dev<Acc> const& devAcc, std::size_t numElements, std::size_t simdWidth)
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    if constexpr(alpaka::simd::isGpuAcc<Acc>())
    {
        // GPU Strategy: 512 threads per block, as many blocks as needed for full coverage
        // - High occupancy for memory latency hiding
        // - Grid-stride loop in kernel handles excess threads gracefully
        constexpr Idx blockSize = 512; // Optimal for memory-bound kernels (16 warps)
        Idx gridBlocks = (numElements + blockSize - 1) / blockSize; // Ceiling division
        if(gridBlocks == 0)
            gridBlocks = 1; // Minimum one block

        std::cout << "[WorkDiv GPU] numElements=" << numElements << " simdWidth=" << simdWidth
                  << " blockSize=" << blockSize << " gridBlocks=" << gridBlocks
                  << " totalThreads=" << (gridBlocks * blockSize) << std::endl;

        return alpaka::WorkDivMembers<Dim, Idx>{gridBlocks, blockSize, Idx(1)};
    }
    else if constexpr(alpaka::simd::isCpuMultiThreadAcc<Acc>())
    {
        // CPU Multi-threaded Strategy: Use hardware concurrency, each thread processes chunks
        // - One thread per physical core to avoid oversubscription
        // - Each thread handles multiple elements for better cache usage
        Idx blockSize = std::thread::hardware_concurrency();
        if(blockSize == 0)
        {
            // Fallback if detection fails
            blockSize = 4;
        }
        // Good cache line utilization
        constexpr Idx elementsPerThread = 64;
        Idx totalWork = numElements;
        Idx gridBlocks = (totalWork + (blockSize * elementsPerThread) - 1) / (blockSize * elementsPerThread);
        if(gridBlocks == 0)
            gridBlocks = 1;

        std::cout << "[WorkDiv CPU-MT] numElements=" << numElements << " simdWidth=" << simdWidth
                  << " hwConcurrency=" << blockSize << " gridBlocks=" << gridBlocks
                  << " totalThreads=" << (gridBlocks * blockSize) << std::endl;

        return alpaka::WorkDivMembers<Dim, Idx>{gridBlocks, blockSize, Idx(1)};
    }
    else
    {
        // CPU Serial Strategy: Single thread with grid-stride for SIMD vectorization
        // - Grid-stride pattern helps with prefetching and SIMD pack processing
        // - Each "block" represents a chunk that benefits from locality
        // Single thread
        constexpr Idx blockSize = 1;
        // SIMD packs needed
        Idx packCount = (numElements + simdWidth - 1) / simdWidth;
        // Multiple chunks for better prefetch (grid-stride loop)
        Idx gridBlocks = std::max(packCount / 8, Idx(1));

        // Note: Alpaka's serial accelerator executes all grid blocks sequentially on a single host thread.
        // gridBlocks therefore represents logical iteration chunks, not parallel hardware threads.
        std::cout << "[WorkDiv CPU-Serial] numElements=" << numElements << " simdWidth=" << simdWidth
                  << " packCount=" << packCount << " gridBlocks=" << gridBlocks << " threadsPerBlock=" << blockSize
                  << " logicalThreads=" << (gridBlocks * blockSize) << " hardwareThreads=1" << std::endl;

        return alpaka::WorkDivMembers<Dim, Idx>{gridBlocks, blockSize, Idx(1)};
    }
}

//! IMPORTANT: SIMD Performance Optimization Requirements
//! For optimal SIMD performance, compile with -O3 optimization and -march=native.
//! performance.

// Define global constants
constexpr uint32_t intScalarR = 11u;
constexpr uint32_t intScalarG = 16u;
constexpr uint32_t intScalarB = 5u;
constexpr uint32_t intScalar32 = 32u;

// Template function for fuzzy equality
template<typename T>
[[maybe_unused]] bool FuzzyEqual(T a, T b)
{
    if constexpr(std::is_floating_point_v<T>)
    {
        return std::fabs(a - b) < (std::numeric_limits<T>::epsilon() * static_cast<T>(100.0));
    }
    else if constexpr(std::is_integral_v<T>)
    {
        return a == b;
    }
    else
    {
        static_assert(
            std::is_floating_point_v<T> || std::is_integral_v<T>,
            "FuzzyEqual<T> is only supported for integral or floating-point types.");
    }
}

// Reference inspiration: Matthias Kretz's CppCon talk on packed RGB SIMD processing:
// https://www.youtube.com/watch?v=LAJ_hywLtMA

// Optimized SIMD kernel (IN-PLACE): overwrites the input pixel buffer with grayscale pixels.
// Fair timing is ensured by reloading the original input data before running the SIMD pass.
class GrayscalePackedSIMDKernel
{
public:
    // In-place kernel: single buffer pointer. Safe because each SIMD pack is fully loaded before being overwritten.
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, uint32_t* data, size_t n) const
    {
        using SimdT = alpaka::simd::PortableSimd<uint32_t, Acc>;
        alpaka::simd::SimdAlgo<Acc, uint32_t> algo(acc, n);

        const SimdT simdMaskFF(0xFFU);
        // Process arrays in SIMD chunks with automatic tail handling
        // base:  Starting index in the array for this SIMD chunk (e.g., 0, 4, 8, 12, ...)
        // valid: Number of valid elements in this chunk (usually simdWidth, but may be less for the final
        // chunk)
        //        For example, if array has 10 elements and simdWidth=4, the chunks would be:
        //        - base=0, valid=4  (elements 0-3)
        //        - base=4, valid=4  (elements 4-7)
        //        - base=8, valid=2  (elements 8-9, final partial chunk)
        algo.forEach(
            [&](std::size_t base, std::size_t valid)
            {
                SimdT vin;
                alpaka::simd::loadSimd<uint32_t, Acc>(data + base, valid, vin);
                auto const A = vin >> 24u;
                auto const R = (vin >> 16u) & simdMaskFF;
                auto const G = (vin >> 8u) & simdMaskFF;
                auto const B = vin & simdMaskFF;
                auto const gray
                    = (R * SimdT(intScalarR) + G * SimdT(intScalarG) + B * SimdT(intScalarB)) / SimdT(intScalar32);
                auto packed = (A << 24u) | (gray << 16u) | (gray << 8u) | gray;
                alpaka::simd::storeSimd<uint32_t, Acc>(packed, valid, data + base);
            });
    }
};

// Scalar(Non-SIMD) kernel: overwrites the input pixel buffer with grayscale pixels.
// Used as baseline; original data is restored before the SIMD kernel for fair comparison.
class GrayscalePackedScalarKernel
{
public:
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, std::uint32_t* data, size_t dataSize) const
    {
        const std::size_t threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const std::size_t numThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        for(std::size_t i = threadIdx; i < dataSize; i += numThreads)
        {
            std::uint32_t p = data[i];
            std::uint32_t a = p >> 24;
            std::uint32_t r = (p >> 16) & 0xFFu;
            std::uint32_t g = (p >> 8) & 0xFFu;
            std::uint32_t b = p & 0xFFu;
            std::uint32_t gray = (r * intScalarR + g * intScalarG + b * intScalarB) / intScalar32;
            data[i] = (a << 24) | (gray << 16) | (gray << 8) | gray;
        }
    }
};

template<typename Acc>
void runPackedRGBAComparison(
    alpaka::Dev<Acc> const& devAcc,
    alpaka::Queue<Acc, alpaka::Blocking>& queue,
    std::vector<std::uint32_t> const& packedRGB,
    std::vector<std::uint32_t>& grayscaleResult,
    size_t elements)
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    const size_t simdWidth = alpaka::simd::simdWidth<std::uint32_t, Acc>();
    std::cout << "simdWidth for packed RGBA (uint32_t) is " << simdWidth << std::endl;

    // Optimized work division based on accelerator type
    auto const workDivPacked = getOptimalWorkDivForRGBA<Acc>(devAcc, elements, simdWidth);
    std::size_t numBlocks = workDivPacked.m_gridBlockExtent[0];
    std::size_t blockSize = workDivPacked.m_blockThreadExtent[0];
    std::size_t totalThreads = numBlocks * blockSize;

    // Allocate single device buffer (in-place processing)
    auto deviceBuffer = alpaka::allocBuf<std::uint32_t, Idx>(devAcc, elements);
    auto* bufferPtr = alpaka::getPtrNative(deviceBuffer);
    bool isAligned = (reinterpret_cast<std::uintptr_t>(bufferPtr) % 32) == 0;
    std::cout << "Device buffer aligned: " << (isAligned ? "YES" : "NO") << std::endl;
    alpaka::memcpy(queue, deviceBuffer, packedRGB);

    float scalarPackedTime = 0.0f;
    float simdPackedTime = 0.0f;

    // Scalar (in-place timing)
    {
        std::cout << "Scalar: " << totalThreads << " threads (" << numBlocks << " blocks × " << blockSize
                  << " threads) simdWidth=" << simdWidth << std::endl;
        // Timing: host wall clock around the kernel launch + completion
        alpaka::wait(queue);
        auto start = std::chrono::high_resolution_clock::now();

        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDivPacked, GrayscalePackedScalarKernel{}, bufferPtr, elements));

        alpaka::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        scalarPackedTime = duration.count() / 1000.0f; // Convert to milliseconds
    }

    // SIMD (in-place timing)
    {
        std::cout << "SIMD  : " << totalThreads << " threads (" << numBlocks << " blocks × " << blockSize
                  << " threads) simdWidth=" << simdWidth << " (in-place)" << std::endl;
        // Restore original input before SIMD for fair comparison
        alpaka::memcpy(queue, deviceBuffer, packedRGB);
        alpaka::wait(queue);
        auto start = std::chrono::high_resolution_clock::now();

        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc>(workDivPacked, GrayscalePackedSIMDKernel{}, bufferPtr, elements));

        alpaka::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        simdPackedTime = duration.count() / 1000.0f; // Convert to milliseconds

        // Copy result back
        alpaka::memcpy(queue, grayscaleResult, deviceBuffer);
    }

    // Results & Speedup Reporting
    float speedup = scalarPackedTime / simdPackedTime;
    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "scalar_time_ms = " << scalarPackedTime << '\n';
    std::cout << "simd_time_ms   = " << simdPackedTime << '\n';
    std::cout << "speedup        = " << speedup << " (scalar/simd)" << '\n';
    std::cout << "simd policy    = " << alpaka::simd::policyName<uint32_t, Acc>() << '\n';
    std::cout << "---------------------------" << std::endl;
}

// Example function - FOCUS ONLY on Matthias Kretz's packed RGBA approach
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, size_t numElements) -> int
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;

    std::cout << "Accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);
    QueueAcc queue(devAcc);

    std::cout << "Testing packed RGBA SIMD processing..." << std::endl;

    // Print SIMD system information
    alpaka::simd::printSimdSystemInfo();

    // Create test data for packed RGBA
    std::vector<std::uint32_t> packedRGBData(numElements);
    std::vector<std::uint32_t> packedGrayscaleResult(numElements);

    // Initialize with sample RGBA values (including alpha channel)
    for(size_t i = 0; i < numElements; ++i)
    {
        // Create some pattern: varying RGBA based on index
        std::uint32_t a = 255; // Full alpha for visibility
        std::uint32_t r = (i * 37) % 256;
        std::uint32_t g = (i * 73) % 256;
        std::uint32_t b = (i * 101) % 256;
        packedRGBData[i] = (a << 24) | (r << 16) | (g << 8) | b; // 0xAARRGGBB format
    }

    runPackedRGBAComparison<Acc>(devAcc, queue, packedRGBData, packedGrayscaleResult, numElements);

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
#ifdef ALPAKA_CI
    size_t numElements = 1 << 20;
#else
    size_t numElements = 1 << 25;
#endif

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg.rfind("numElements=", 0) == 0)
        {
            try
            {
                numElements = std::stoul(arg.substr(12));
            }
            catch(...)
            {
                std::cerr << "Invalid numElements value" << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    if(numElements == 0)
    {
        std::cerr << "numElements must be > 0" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Elements: " << numElements;
    if((numElements & (numElements - 1)) != 0)
    {
        std::cout << " (not a power of two)";
    }
    std::cout << std::endl;

    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag, numElements); });
}
