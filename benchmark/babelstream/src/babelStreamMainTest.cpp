
#include "babelStreamCommon.hpp"
#include "catch2/catch_session.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iterator>
#include <string>

using namespace alpaka;

/**
 * Babelstream benchmarking example. Babelstream has 5 kernels. Add, Multiply, Copy, Triad and Dot. NStream is
 * optional. Init kernel is run before 5 standard kernel sequence. Babelstream is a memory-bound benchmark since the
 * main operation in the kernels has high Code Balance (bytes/FLOP) value. For example c[i] = a[i] + b[i]; has 2 reads
 * 1 writes and has one FLOP operation. For double precision each read-write is 8 bytes. Hence Code Balance (3*8 / 1) =
 * 24 bytes/FLOP.
 *
 * Some implementations and the documents are accessible through https://github.com/UoB-HPC
 *
 * Can be run with custom arguments as well as catch2 arguments
 * Run with Custom arguments and for kernels: init, copy, mul, add, triad (and dot kernel if a multi-thread acc
 * available):
 * ./babelstream --array-size=33554432 --number-runs=100
 * Run with Custom arguments and select from 3 kernel groups: all, triad, nstream
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=triad (only triad kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=nstream (only nstream kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=all (default case. Add, Multiply, Copy, Triad
 * and Dot) Run with default array size and num runs:
 * ./babelstream
 * Run with Catch2 arguments and default array size and num runs:
 * ./babelstream --success
 * ./babelstream -r xml
 * Run with Custom and catch2 arguments together:
 * ./babelstream  --success --array-size=1280000 --number-runs=10
 * Help to list custom and catch2 arguments
 * ./babelstream -?
 * ./babelstream --help
 *  According to tests, 2^25 or larger data size values are needed for proper benchmarking:
 *  ./babelstream --array-size=33554432 --number-runs=100
 */

// Main function that integrates Catch2 and custom argument handling
int main(int argc, char* argv[])
{
    // Handle custom arguments
    handleCustomArguments(argc, argv);

    // Initialize Catch2 and pass the command-line arguments to it
    int result = Catch::Session().run(argc, argv);

    // Return the result of the tests
    return result;
}

struct SimdForEachKernel
{
    //! \param acc The accelerator to be executed on.
    //! \param func functor applied to each SIMD package.
    //! \param arg0 MdSpan from which the problem size is derived
    //! \param args MdSpan other spans
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::IDataSource auto arg0,
        alpaka::concepts::IDataSource auto... args) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(acc, arg0.getExtents(), func, arg0, args...);
    }
};

struct SimdInitOp
{
    constexpr void operator()(auto const&, auto a, auto b, auto c) const
    {
        using SimdType = ALPAKA_TYPEOF(a.load());
        a = SimdType::fill(initA);
        b = SimdType::fill(initB);
        c = SimdType::fill(initC);
    }
};

struct SimdCopyOp
{
    constexpr void operator()(auto const&, auto const a, auto c) const
    {
        c = a.load();
    }
};

struct SimdMultOp
{
    constexpr void operator()(auto const&, auto b, auto const c) const
    {
        using T = GetValueType_t<ALPAKA_TYPEOF(b)>;
        T const scalar = static_cast<T>(scalarVal);
        b = scalar * c.load();
    }
};

struct SimdAddOp
{
    constexpr void operator()(auto const&, auto const a, auto b, auto c) const
    {
        c = a.load() + b.load();
    }
};

struct SimdTriadOp
{
    constexpr void operator()(auto const&, auto a, auto const b, auto const c) const
    {
        using T = GetValueType_t<ALPAKA_TYPEOF(a)>;
        T const scalar = static_cast<T>(scalarVal);
        a = b.load() + scalar * c.load();
    }
};

struct SimdNStreamOp
{
    constexpr void operator()(auto const&, auto a, auto const b, auto const c) const
    {
        using T = GetValueType_t<ALPAKA_TYPEOF(a)>;
        T const scalar = static_cast<T>(scalarVal);
        a = a.load() + b.load() + scalar * c.load();
    }
};

struct DotMultiplication
{
    constexpr auto operator()(concepts::Simd auto&& simdA, concepts::Simd auto&& simdB) const
    {
        return simdA * simdB;
    }
};

//! \brief The Function for testing babelstream kernels for given Acc type and data type.
//! \tparam TAcc the accelerator type
//! \tparam T_DataType The data type to differentiate single or double data type based tests.
template<typename T_DataType>
void testKernels(auto const deviceSpec, auto const exec)
{
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        std::cout << "Kernels: Init, Copy, Mul, Add, Triad, Dot Kernels" << '\n';
    }

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device devAcc = devSelector.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
    // support for double precision is not guaranteed for sycl devices such as Intel GPUs
    if constexpr(std::is_same_v<T_DataType, double> && deviceSpec.getApi() == api::oneApi)
    {
        if(devAcc.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
        {
            std::cout << onHost::getName(devAcc) << " does not support double precision"
                      << "\n";
            std::cout << "Skip benchmark.\n";
            std::cout << "For Intel Arc GPUs, use the environemnt variables `IGC_EnableDPEmulation=1 "
                         "OverrideDefaultFP64Settings=1` to emulate double precision support.\n";
            return;
        }
    }
#endif

    std::cout << devAcc.getDeviceProperties() << std::endl;

    std::cout << "executor: " << onHost::demangledName(exec) << std::endl;
    std::cout << "deviceKind: " << devAcc.getDeviceKind().getName() << std::endl;

    // A MetaData class instance to keep the benchmark info and results to print later. Does not include intermediate
    // runtime data.
    BenchmarkMetaData metaData;

    // Convert data-type to string to display
    std::string dataTypeStr;
    if(std::is_same<T_DataType, float>::value)
    {
        dataTypeStr = "single";
    }
    else if(std::is_same<T_DataType, double>::value)
    {
        dataTypeStr = "double";
    }

    // Get the host device for allocating memory on the host
    onHost::Queue queue = devAcc.makeQueue();

    // Create vectors
    using Idx = std::uint32_t;
    auto arraySize = static_cast<Idx>(arraySizeMain);

    // Acc buffers
    concepts::IBuffer auto bufAccInputA = onHost::alloc<T_DataType>(devAcc, arraySize);
    concepts::IBuffer auto bufAccInputB = onHost::allocLike(devAcc, bufAccInputA);
    concepts::IBuffer auto bufAccOutputC = onHost::allocLike(devAcc, bufAccInputA);

    // Host buffer as the result
    concepts::IBuffer auto bufHostOutputA = onHost::allocHostLike(bufAccInputA);
    concepts::IBuffer auto bufHostOutputB = onHost::allocHostLike(bufAccInputB);
    concepts::IBuffer auto bufHostOutputC = onHost::allocHostLike(bufAccOutputC);

    /* Each frame will have 64 elements processed by each thread.
     * The number of frames is calculated based on the array size and the number of elements processed by each thread.
     *
     * @todo The value is currently a magic number but should be derived from the SIMD width of the device and a factor
     * to reflect the instruction level parallelism. This is currently not well abstracted in alpaka and requires that
     * a kernel can reflect the concurrency bytes used for the `SimdAlgo::concurrent()` back to the host, e.g. some
     * as we use for dynamic shared memory.
     */
    uint32_t elementsPerFrameItem = getNumElemPerThread<T_DataType>(queue);

    auto numFrames = divExZero(arraySize, static_cast<Idx>(blockThreadExtentMain) * elementsPerFrameItem);
    auto dataBlocking = onHost::FrameSpec{numFrames, static_cast<Idx>(blockThreadExtentMain), exec};

    // To record runtime data generated while running the kernels
    RuntimeResults runtimeResults;

    // Lambda for measuring run-time
    auto measureKernelExec = [&](auto&& kernelFunc, [[maybe_unused]] auto&& kernelLabel)
    {
        double runtime = 0.0;
        onHost::wait(queue);
        auto start = std::chrono::high_resolution_clock::now();
        kernelFunc();
        onHost::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // get duration in seconds
        std::chrono::duration<double> duration = end - start;
        runtime = duration.count();
        runtimeResults.kernelToRundataMap[kernelLabel]->timingsSuccessiveRuns.push_back(runtime);
    };


    // Initialize logger before running kernels
    // Runtime result initialisation to be filled by each kernel
    runtimeResults.addKernelTimingsVec("InitKernel");
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        runtimeResults.addKernelTimingsVec("CopyKernel");
        runtimeResults.addKernelTimingsVec("AddKernel");
        runtimeResults.addKernelTimingsVec("TriadKernel");
        runtimeResults.addKernelTimingsVec("MultKernel");
        runtimeResults.addKernelTimingsVec("DotKernel");
    }
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        runtimeResults.addKernelTimingsVec("NStreamKernel");
    }
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        runtimeResults.addKernelTimingsVec("TriadKernel");
    }

    // Init kernel
    measureKernelExec(
        [&]()
        {
            queue.enqueue(
                dataBlocking,
                KernelBundle{SimdForEachKernel{}, SimdInitOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
        },
        "InitKernel");

    // Init kernel will be run for all cases therefore add it to metadata unconditionally
    metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);

    // Dot kernel result
    T_DataType resultDot = static_cast<T_DataType>(0.0);

    // Main for loop to run the kernel-sequence
    for(auto i = 0; i < numberOfRuns; i++)
    {
        if(kernelsToBeExecuted == KernelsToRun::All)
        {
            // Test the copy-kernel. Copy A one by one to C.
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        dataBlocking,
                        KernelBundle{SimdForEachKernel{}, SimdCopyOp{}, bufAccInputA, bufAccOutputC});
                },
                "CopyKernel");

            // Test the scaling-kernel. Calculate B=scalar*C. Where C = A.
            measureKernelExec(
                [&]() { queue.enqueue(dataBlocking, SimdForEachKernel{}, SimdMultOp{}, bufAccInputB, bufAccOutputC); },
                "MultKernel");

            // Test the addition-kernel. Calculate C=A+B. Where B=scalar*C or B=scalar*A.
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        dataBlocking,
                        KernelBundle{SimdForEachKernel{}, SimdAddOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
                },
                "AddKernel");
        }
        // Triad kernel is run for 2 command line arguments
        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Triad)
        {
            // Test the Triad-kernel. Calculate A=B+scalar*C. Where C is A+scalar*A.
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        dataBlocking,
                        KernelBundle{SimdForEachKernel{}, SimdTriadOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
                },
                "TriadKernel");
        }
        if(kernelsToBeExecuted == KernelsToRun::All)
        {
            // Vector of sums of each block
            auto bufAccSumPerBlock = onHost::allocDeferred<T_DataType>(queue, 1u);
            auto bufHostSumPerBlock = onHost::allocHostLike(bufAccSumPerBlock);

            // Test Dot kernel with specific blocksize which is larger than one
            measureKernelExec(
                [&]()
                {
                    onHost::transformReduce(
                        queue,
                        exec,
                        T_DataType{0},
                        bufAccSumPerBlock,
                        std::plus{},
                        DotMultiplication{},
                        bufAccInputA,
                        bufAccInputB);
                    onHost::memcpy(queue, bufHostSumPerBlock, bufAccSumPerBlock);
                    onHost::wait(queue);
                    resultDot = bufHostSumPerBlock[0u];
                },
                "DotKernel");
        }
        // NStream kernel is run only for one command line argument
        if(kernelsToBeExecuted == KernelsToRun::NStream)
        {
            // Test the NStream-kernel. Calculate A += B + scalar * C;
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        dataBlocking,
                        KernelBundle{SimdForEachKernel{}, SimdNStreamOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
                },
                "NStreamKernel");
        }
        onHost::wait(queue);
    } // End of MAIN LOOP which runs the kernels many times


    // Copy results back to the host, measure copy time
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Copy arrays back to host since the execution of kernels except dot kernel finished
        onHost::memcpy(queue, bufHostOutputC, bufAccOutputC);
        onHost::memcpy(queue, bufHostOutputB, bufAccInputB);
        onHost::memcpy(queue, bufHostOutputA, bufAccInputA);
        onHost::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // Get duration in seconds
        std::chrono::duration<double> duration = end - start;
        double copyRuntime = duration.count();
        metaData.setItem(BMInfoDataType::CopyTimeFromAccToHost, copyRuntime);
    }

    //
    // Result Verification and BW Calculation for 3 cases
    //

    // Generated expected values by doing the same chain of operations due to floating point error
    T_DataType expectedA = static_cast<T_DataType>(initA);
    T_DataType expectedB = static_cast<T_DataType>(initB);
    T_DataType expectedC = static_cast<T_DataType>(initC);

    // To calculate expected results by applying at host the same operation sequence
    calculateBabelstreamExpectedResults(expectedA, expectedB, expectedC);

    // Verify the resulting data, if kernels are init, copy, mul, add, triad and dot kernel
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        // Find sum of the errors as sum of the differences from expected values
        constexpr T_DataType initVal{static_cast<T_DataType>(0.0)};
        T_DataType sumErrC{initVal}, sumErrB{initVal}, sumErrA{initVal};

        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrC += std::fabs(std::data(bufHostOutputC)[static_cast<Idx>(i)] - expectedC);
            sumErrB += std::fabs(std::data(bufHostOutputB)[static_cast<Idx>(i)] - expectedB);
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }

        // Normalize and compare sum of the errors
        // Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(sumErrC / static_cast<T_DataType>(arraySize), static_cast<T_DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrB / static_cast<T_DataType>(arraySize), static_cast<T_DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrA / static_cast<T_DataType>(arraySize), static_cast<T_DataType>(0.0)));
        onHost::wait(queue);

        // Verify Dot kernel
        T_DataType const expectedSum = static_cast<T_DataType>(arraySize) * expectedA * expectedB;

        // allow a summation error of 100 epsilon per 32 mega elements
        // The original bablesteam is always checking 8 digits only.
        Idx presicionBaseArraySize = 32 * 1024 * 1024;
        T_DataType epsScaling = divExZero(arraySize, presicionBaseArraySize);
        //  Dot product should be identical to arraySize*valA*valB
        //  Use a different equality check if floating point errors exceed the precision of FuzzyEqual function
        REQUIRE(
            FuzzyEqual<T_DataType>(
                std::fabs(resultDot - expectedSum) / expectedSum,
                static_cast<T_DataType>(0.0),
                static_cast<T_DataType>(100.0) * epsScaling));

        // Set workdivs of benchmark metadata to be displayed at the end
        metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivCopy, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivAdd, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivMult, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);
    }
    // Verify the Triad Kernel result if "--run-kernels=triad".
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        // Verify triad by summing the error
        auto sumErrA = static_cast<T_DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }

        REQUIRE(FuzzyEqual(sumErrA / static_cast<T_DataType>(arraySize) / expectedA, static_cast<T_DataType>(0.0)));
        metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);
    }
    // Verify the NStream Kernel result if "--run-kernels=nstream".
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        auto sumErrA = static_cast<T_DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }
        REQUIRE(FuzzyEqual(sumErrA / static_cast<T_DataType>(arraySize) / expectedA, static_cast<T_DataType>(0.0)));

        metaData.setItem(BMInfoDataType::WorkDivNStream, dataBlocking);
    }

    // Runtime results of the benchmark: Calculate throughput and bandwidth
    // Set throuput values depending on the kernels
    runtimeResults.initializeByteReadWrite<T_DataType>(arraySize);
    runtimeResults.calculateBandwidthsForKernels<T_DataType>();

    // Set metadata to display all benchmark related information.
    //
    // All information about benchmark and results are stored in a single map
    metaData.setItem(BMInfoDataType::TimeStamp, getCurrentTimestamp());
    metaData.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    metaData.setItem(BMInfoDataType::DataSize, std::to_string(arraySizeMain));
    metaData.setItem(BMInfoDataType::DataType, dataTypeStr);
    // Device and accelerator
    metaData.setItem(BMInfoDataType::DeviceName, onHost::getName(devAcc));
    metaData.setItem(BMInfoDataType::AcceleratorType, onHost::demangledName(exec));
    // XML reporter of catch2 always converts to Nano Seconds
    metaData.setItem(BMInfoDataType::TimeUnit, "Nano Seconds");

    // get labels from the map
    std::vector<std::string> kernelLabels;
    std::transform(
        runtimeResults.kernelToRundataMap.begin(),
        runtimeResults.kernelToRundataMap.end(),
        std::back_inserter(kernelLabels),
        [](auto const& pair) { return pair.first; });
    // Join elements and create a comma separated string and set item
    metaData.setItem(BMInfoDataType::KernelNames, joinElements(kernelLabels, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> values(runtimeResults.getThroughputKernelArray());
    metaData.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(values, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> valuesBW(runtimeResults.getBandwidthKernelVec());
    metaData.setItem(BMInfoDataType::KernelBandwidths, joinElements(valuesBW, ", "));

    metaData.setItem(BMInfoDataType::KernelMinTimes, joinElements(runtimeResults.getMinExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelMaxTimes, joinElements(runtimeResults.getMaxExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelAvgTimes, joinElements(runtimeResults.getAvgExecTimeKernelArray(), ", "));
    // Print the summary as a table, if a standard serialization is needed other functions of the class can be used
    std::cout << metaData.serializeAsTable() << '\n';
}

using Backends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Float>", "[benchmark-test]", Backends)
{
    auto backend = TestType::makeDict();
    // Run tests for the float data type
    testKernels<float>(alpaka::onHost::DeviceSpec{backend}, alpaka::getExecutor(backend));
}

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Double>", "[benchmark-test]", Backends)
{
    auto backend = TestType::makeDict();
    // Run tests for the double data type
    testKernels<double>(alpaka::onHost::DeviceSpec{backend}, alpaka::getExecutor(backend));
}
