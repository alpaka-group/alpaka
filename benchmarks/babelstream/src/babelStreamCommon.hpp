#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
    // Default array size, can be changed from command line arguments.
    // To display cmd line args use ./babelstream --help or -?
    // According to tests, 2^25 or larger values are needed for proper benchmarking:
    // ./babelstream --array-size=33554432 --number-runs=100
    // To prevent timeouts in CI, a smaller default value is used.
    [[maybe_unused]] auto arraySizeMain = 1024 * 256;

    // Minimum array size to be used.
    [[maybe_unused]] constexpr auto minArrSize = 1024 * 256;

    // Scalar value for Mul and Triad kernel parameters.
    [[maybe_unused]] constexpr double scalarVal = 0.4;

    // Block thread extent for DotKernel test work division parameters.
    [[maybe_unused]] constexpr auto blockThreadExtentMain = 1024;
    [[maybe_unused]] constexpr auto dotGridBlockExtent = 256;

    // Number of runs for each kernel, can be changed by command line arguments.
    // At least 100 runs are recommended for good benchmarking.
    // To prevent timeouts in CI, a small value is used.
    [[maybe_unused]] auto numberOfRuns = 2;

    // Data input values for babelstream.
    [[maybe_unused]] constexpr double initA = 0.1;
    [[maybe_unused]] constexpr double initB = 0.2;
    // Change this if triad kernel is going to be run alone
    [[maybe_unused]] constexpr double initC = 0.0;

    //! Values corresponding to the command line argument run-kernels
    enum class KernelsToRun
    {
        All, // init, add, copy, mul, triad, dot
        Triad, // only init and triad
        NStream // only init and nstream
    };

    // Define the variable showing the kernel(s) being run
    [[maybe_unused]] KernelsToRun kernelsToBeExecuted{KernelsToRun::All};

    //! handleCustomArguments Gets custom cmd line arguments from the all arguments.
    //! Namely gets --array-size=1234 and --number-runs=1234 and keeps the others which are
    //! command line args for Catch2 session.
    [[maybe_unused]] static void handleCustomArguments(int& argc, char* argv[])
    {
        std::vector<char*> newArgv({argv[0]}); // keep program name

        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if(arg.rfind("--array-size=", 0) == 0)
            {
                try
                {
                    // Convert argument to integer
                    auto arrSize = std::stoi(arg.substr(13));
                    if(arrSize > minArrSize)
                    {
                        arraySizeMain = arrSize;
                        std::cout << "Array size set to: " << arraySizeMain << std::endl;
                    }
                    else
                    {
                        std::cout << "Array size too small. Must be at least " << minArrSize
                                  << ", using default: " << arraySizeMain << std::endl;
                    }
                }
                catch(std::invalid_argument const&)
                {
                    std::cerr << "Invalid array size argument: " << arg << ". Default value used." << std::endl;
                }
            }
            else if(arg.rfind("--number-runs=", 0) == 0)
            {
                try
                {
                    // Convert argument to integer
                    auto const numRuns = std::stoi(arg.substr(14));
                    if(numRuns > 0)
                    {
                        numberOfRuns = numRuns;
                        std::cout << "Number of runs provided: " << numberOfRuns << std::endl;
                    }
                    else
                    {
                        std::cout << "Using default number of runs: " << numberOfRuns << std::endl;
                    }
                }
                catch(std::invalid_argument const&)
                {
                    std::cerr << "Invalid number of runs argument: " << arg << " . Default value used." << std::endl;
                }
            }
            else if(arg.rfind("--run-kernels=", 0) == 0)
            {
                // Get argument to determine which kernels will be run
                auto const kernelsString = arg.substr(14);
                if(kernelsString == "nstream")
                {
                    std::cout << "Only nstream kernel will be executed." << std::endl;
                    kernelsToBeExecuted = KernelsToRun::NStream;
                }
                else if(kernelsString == "triad")
                {
                    kernelsToBeExecuted = KernelsToRun::Triad;
                    std::cout << "Only triad kernel will be executed." << std::endl;
                }
                else if(kernelsString == "all")
                {
                    // The variable kernelsToBeExecuted default value is  "all";
                    kernelsToBeExecuted = KernelsToRun::All;
                    std::cout << "All 5 babelstream kernels are going to be executed." << std::endl;
                }
            }
            else
            {
                // If it's not a custom argument, keep it for Catch2
                newArgv.push_back(argv[i]);
            }
            if(arg.rfind("-?", 0) == 0 || arg.rfind("--help", 0) == 0 || arg.rfind("-h", 0) == 0)
            {
                std::cout << "Usage of custom arguments (arguments which are not Catch2):  --array-size=33554432 and "
                             "--number-runs=100\n"
                          << std::endl;
                std::cout << "If you want to run only nstream kernel or triad kernel use --run-kernels=nstream or  "
                             "--run-kernels=triad. Otherwise all 5 standard kernels will be executed. Copy, Mul, Add, "
                             "Triad and Dot kernel."
                          << std::endl;
            }
        }

        // Update argc and argv to exclude custom arguments
        argc = static_cast<int>(newArgv.size());
        for(int i = 0; i < argc; ++i)
        {
            argv[i] = newArgv[static_cast<size_t>(i)];
        }

        // Array size must a multiple of
        if(arraySizeMain % blockThreadExtentMain != 0)
            throw std::runtime_error(
                "Array size is " + std::to_string(arraySizeMain) + ". It must be a multiple of block-size "
                + std::to_string(blockThreadExtentMain));
    }

    //! FuzzyEqual compares two floating-point or integral type values.
    //! \tparam T Type of the values to compare.
    //! \param a First value to compare.
    //! \param b Second value to compare.
    //! \return Returns true if the values are approximately equal (for floating-point types) or exactly equal (for
    //! integral types).
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

    //!   Gets the current timestamp and returns it as a string.
    //! \return A string representation of the current timestamp in the format "YYYY-MM-DD HH:MM:SS".
    [[maybe_unused]] static std::string getCurrentTimestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %X");
        return ss.str();
    }

    //! joinElements  Joins the elements of a vector into a string, separated by a specified delimiter.
    //! \tparam T Type of the elements in the vector.
    //! \param vec The vector of elements to join.
    //! \param delim The delimiter to separate the elements in the resulting string.
    //! \return A string with the vector elements separated by the specified delimiter.
    template<typename T>
    [[maybe_unused]] static std::string joinElements(std::vector<T> const& vec, std::string const& delim)
    {
        return std::accumulate(
            vec.begin(),
            vec.end(),
            std::string(),
            [&delim](std::string const& a, T const& b)
            {
                std::ostringstream oss;
                if(!a.empty())
                    oss << a << delim;
                oss << std::setprecision(5) << b;
                return oss.str();
            });
    }

    //! findMinMax  Finds the minimum and maximum elements in a container.
    //! \tparam Container The type of the container.
    //! \param times The container from which to find the minimum and maximum elements.
    //! \return A pair containing the minimum and maximum values in the container.
    //! \note The first element is omitted if the container size is larger than 1, as the result is used in time
    //! measurement for benchmarking.
    template<typename Container>
    [[maybe_unused]] static auto findMinMax(Container const& times)
        -> std::pair<typename Container::value_type, typename Container::value_type>
    {
        if(times.empty())
            return std::make_pair(typename Container::value_type{}, typename Container::value_type{});

        // Default to min and max being the same element for single element containers
        auto minValue = *std::min_element(times.begin(), times.end());
        auto maxValue = minValue;

        if(times.size() > 1)
        {
            // Calculate min and max ignoring the first element
            minValue = *std::min_element(times.begin() + 1, times.end());
            maxValue = *std::max_element(times.begin() + 1, times.end());
        }

        return std::make_pair(minValue, maxValue);
    }

    //! findAverage  Calculates the average value of elements in a container, does not take into account the first one.
    //! \tparam Container The type of the container.
    //! \param elements The container from which to calculate the average.
    //! \return The average value of the elements in the container without considering the first element.
    template<typename Container>
    [[maybe_unused]] static auto findAverage(Container const& elements) -> typename Container::value_type
    {
        if(elements.empty())
            return typename Container::value_type{};

        if(elements.size() == 1)
            return elements.front(); // Only one element, return it as the average

        // Calculate the sum of the elements, start from the second one
        auto sum = std::accumulate(elements.begin() + 1, elements.end(), typename Container::value_type{});

        // Calculate and return the average, take into account that one element is not used
        return sum / static_cast<typename Container::value_type>(elements.size() - 1);
    }

    //!   Enum class representing benchmark information data types.
    enum class BMInfoDataType
    {
        AcceleratorType,
        TimeStamp,
        NumRuns,
        DataSize,
        DataType,
        CopyTimeFromAccToHost,
        WorkDivInit,
        WorkDivCopy,
        WorkDivAdd,
        WorkDivTriad,
        WorkDivMult,
        WorkDivDot,
        WorkDivNStream,
        DeviceName,
        TimeUnit,
        KernelNames,
        KernelBandwidths,
        KernelDataUsageValues,
        KernelMinTimes,
        KernelMaxTimes,
        KernelAvgTimes
    };

    //! typeToTypeStr Converts BMInfoDataType enum values to their corresponding string representations.
    //! \param item The BMInfoDataType enum type value to convert to a more explicit string with units.
    //! \return A string representation of the given BMInfoDataType enum value.
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wswitch-default"
#    pragma clang diagnostic ignored "-Wcovered-switch-default"
#endif
    static std::string typeToTypeStr(BMInfoDataType item)
    {
        switch(item)
        {
        case BMInfoDataType::AcceleratorType:
            return "AcceleratorType";
        case BMInfoDataType::TimeStamp:
            return "TimeStamp";
        case BMInfoDataType::NumRuns:
            return "NumberOfRuns";
        case BMInfoDataType::DataSize:
            return "DataSize(items)";
        case BMInfoDataType::DataType:
            return "Precision";
        case BMInfoDataType::CopyTimeFromAccToHost:
            return "AccToHost Memcpy Time(sec)";
        case BMInfoDataType::DeviceName:
            return "DeviceName";
        case BMInfoDataType::TimeUnit:
            return "TimeUnitForXMLReport";
        case BMInfoDataType::KernelNames:
            return "Kernels";
        case BMInfoDataType::KernelDataUsageValues:
            return "DataUsage(MB)";
        case BMInfoDataType::KernelBandwidths:
            return "Bandwidths(GB/s)";
        case BMInfoDataType::KernelMinTimes:
            return "MinTime(s)";
        case BMInfoDataType::KernelMaxTimes:
            return "MaxTime(s)";
        case BMInfoDataType::KernelAvgTimes:
            return "AvgTime(s)";
        case BMInfoDataType::WorkDivInit:
            return "WorkDivInit ";
        case BMInfoDataType::WorkDivCopy:
            return "WorkDivCopy ";
        case BMInfoDataType::WorkDivAdd:
            return "WorkDivAdd  ";
        case BMInfoDataType::WorkDivTriad:
            return "WorkDivTriad";
        case BMInfoDataType::WorkDivMult:
            return "WorkDivMult ";
        case BMInfoDataType::WorkDivDot:
            return "WorkDivDot  ";
        case BMInfoDataType::WorkDivNStream:
            return "WorkDivNStream";
        default:
            return "";
        }
    }
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif
    //! getDataThroughput Calculates the data throughput for processing the entire array.
    //! \tparam DataType The type of the data.
    //! \tparam T The type of the parameters.
    //! \param readsWrites The number of read/write operations.
    //! \param arraySize The size of the array.
    //! \return The calculated data throughput in MB.
    template<typename DataType, typename T>
    [[maybe_unused]] static double getDataThroughput(T readsWrites, T arraySize)
    {
        auto throughput = readsWrites * sizeof(DataType) * arraySize;
        // convert to MB (not MiB)
        return static_cast<double>(throughput) * 1.0E-6;
    }

    //! calculateBandwidth Calculates the bandwidth in GB/sec.
    //! \tparam T The type of bytesReadWriteMB.
    //! \tparam U The type of runTimeSeconds (e.g., double).
    //! \param bytesReadWriteMB The amount of data read/write in MB.
    //! \param runTimeSeconds The runtime in seconds.
    //! \return The calculated bandwidth in GB/sec.
    template<typename T, typename U>
    [[maybe_unused]] static double calculateBandwidth(T bytesReadWriteMB, U runTimeSeconds)
    {
        // Divide by 1.0E+3 to convert from MB to GB (not GiB)
        auto bytesReadWriteGB = static_cast<double>(bytesReadWriteMB) * (1.0E-3);
        return bytesReadWriteGB / static_cast<double>(runTimeSeconds);
    }

    //! \brief calculateBabelstreamExpectedResults Fins the expected values by doing the same operations with the
    //! initial values \param expectedA Array item value expected (all array values are equal in original babelstream)
    //! \param expectedB Array item value expected (all array values are equal in original babelstream)
    //! \param expectedC Array item value expected (all array values are equal in original babelstream)
    template<typename T>
    [[maybe_unused]] static void calculateBabelstreamExpectedResults(T& expectedA, T& expectedB, T& expectedC)
    {
        //
        // All items of arrays are the same, therefore an expected value is for the whole array
        for(auto i = 0; i < numberOfRuns; i++)
        {
            if(kernelsToBeExecuted == KernelsToRun::All)
            {
                expectedC = expectedA;
                expectedB = static_cast<T>(scalarVal) * expectedC;
                expectedC = expectedA + expectedB;
                expectedA = expectedB + static_cast<T>(scalarVal) * expectedC;
            }
            else if(kernelsToBeExecuted == KernelsToRun::Triad)
            {
                expectedA = expectedB + static_cast<T>(scalarVal) * expectedC;
            }
            else if(kernelsToBeExecuted == KernelsToRun::NStream)
            {
                // each run changes the result
                expectedA += expectedB + static_cast<T>(scalarVal) * expectedC;
            }
        }
    }

    /**
     * /brief The RuntimeResults class bundles the kernel runtime data in a map
     * The keys of the map are kernel names the values of the map are KernelRunData struct pointers
     */
    class RuntimeResults
    {
        struct KernelRunData
        {
            std::vector<double> timingsSuccessiveRuns; // Stores execution timings of successive runs
            double byteReadWriteMB{0}; // Bytes read/write in MB
            double bandwidthKernel{0}; // Bandwidth of kernel
            double minExecTime{0}; // Minimum execution time
            double maxExecTime{0}; // Maximum execution time
            double avgExecTime{0}; // Average execution time
        };

    public:
        // Map from kernelName (string) to a unique_ptr for KernelRunData
        // Using unique_ptr for automatic memory management
        std::map<std::string, std::unique_ptr<KernelRunData>> kernelToRundataMap;

        // Function to initialize the byteReadWriteMB field for each kernel
        template<typename DataType>
        void initializeByteReadWrite(size_t arraySize)
        {
            // Define kernel names and their throughput values based on the provided array size
            std::map<std::string, double> throughputValues
                = {{"InitKernel", getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize))},
                   {"CopyKernel", getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize))},
                   {"MultKernel", getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize))},
                   {"AddKernel", getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize))},
                   {"TriadKernel", getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize))},
                   {"DotKernel", getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize))},
                   {"NStreamKernel", getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize))}};

            // Populate each KernelRunData entry in kernelToRundataMap
            for(auto const& [kernelName, throughput] : throughputValues)
            {
                // Check if the kernel name exists in the map
                if(kernelToRundataMap.find(kernelName) != kernelToRundataMap.end())
                {
                    // Set the byteReadWriteMB field in the corresponding KernelRunData
                    kernelToRundataMap[kernelName]->byteReadWriteMB = throughput;
                }
            }
        }

        //! /brief calculateBandwidthsForKernels Function calculates bandwidth for each kernel and update execution
        //! times Fills the fields of bandwidth, execution min-time, execution max-time and execution avg-time
        template<typename DataType>
        void calculateBandwidthsForKernels()
        {
            for(auto const& [kernelName, kernelData] : kernelToRundataMap)
            {
                // Calculate min and max execution times from recorded vector of times for the kernel named kernelName
                auto const minmaxPair = findMinMax(kernelData->timingsSuccessiveRuns);
                kernelData->minExecTime = minmaxPair.first;
                kernelData->maxExecTime = minmaxPair.second;

                // Calculate average execution time
                kernelData->avgExecTime = findAverage(kernelData->timingsSuccessiveRuns);

                // Calculate bandwidth based on byteReadWriteMB and min execution time
                kernelData->bandwidthKernel = calculateBandwidth(kernelData->byteReadWriteMB, minmaxPair.first);
            }
        }

        //! /brief  Get item from each struct in the map and make a vector
        //! /tparam Func is the accessor function to access to a specific field
        template<typename Func>
        std::vector<double> getItemFromStructs(Func accessor) const noexcept
        {
            std::vector<double> results;
            for(auto const& [key, dataStruct] : kernelToRundataMap)
            {
                results.push_back(accessor(dataStruct.get())); // Dereference unique_ptr with .get()
            }
            return results;
        }

        // Functions to retrieve specific fields as vectors for all kernels
        std::vector<double> getBandwidthKernelVec() const noexcept
        {
            return getItemFromStructs([](KernelRunData const* dataStruct) { return dataStruct->bandwidthKernel; });
        }

        std::vector<double> getThroughputKernelArray() const noexcept
        {
            return getItemFromStructs([](KernelRunData const* dataStruct) { return dataStruct->byteReadWriteMB; });
        }

        std::vector<double> getAvgExecTimeKernelArray() const noexcept
        {
            return getItemFromStructs([](KernelRunData const* dataStruct) { return dataStruct->avgExecTime; });
        }

        std::vector<double> getMinExecTimeKernelArray() const noexcept
        {
            return getItemFromStructs([](KernelRunData const* dataStruct) { return dataStruct->minExecTime; });
        }

        std::vector<double> getMaxExecTimeKernelArray() const noexcept
        {
            return getItemFromStructs([](KernelRunData const* dataStruct) { return dataStruct->maxExecTime; });
        }

        // Function to add a kernelName-timesForRuns pair by storing a unique_ptr to KernelRunData
        void addKernelTimingsVec(std::string const& kernelName) noexcept
        {
            // Use make_unique to create a new KernelRunData object and store it in the map
            kernelToRundataMap[kernelName] = std::make_unique<KernelRunData>();
        }
    };

    //! MetaData class to store and serialize benchmark information.
    //! \details The MetaData class includes a single map to keep all benchmark information and provides serialization
    //! methods for generating output.
    class BenchmarkMetaData
    {
    private:
        // Information type to string. String can be comma separated values.
        std::map<BMInfoDataType, std::string> metaDataMap;

    public:
        //! setItem  Sets an item in the metadata map.
        //! \tparam T The type of the value to store.
        //! \param key The BMInfoDataType key.
        //! \param value The value to store associated with the key.
        template<typename T>
        [[maybe_unused]] void setItem(BMInfoDataType key, T const& value)
        {
            std::ostringstream oss;
            oss << value;
            metaDataMap[key] = oss.str();
        }

        //! serialize  Serializes the entire metadata to a string.
        //! \return A string containing the serialized metadata.
        //! \details This is standard serialization and produces output that can be post-processed easily.
        [[maybe_unused]] std::string serialize() const
        {
            std::stringstream ss;
            for(auto const& pair : metaDataMap)
            {
                ss << "\n" << typeToTypeStr(pair.first) << ":" << pair.second;
            }
            return ss.str();
        }

        //! serializeAsTable Serializes the metadata into a more structured format for easy visual inspection.
        //! \return A string containing the serialized metadata as a table.
        //! \details The method first serializes general information, then creates a summary as a table where each row
        //! represents a kernel.
        [[maybe_unused]] std::string serializeAsTable() const
        {
            std::stringstream ss;
            // define lambda to add values to a string stream created already
            auto addItemValueToSS = [&, this](BMInfoDataType item)
            {
                if(metaDataMap.count(item) != 0)
                    ss << "\n" << typeToTypeStr(item) << ":" << metaDataMap.at(item);
            };

            // Initially choose some data to serialize from the meta-data map to add to string stream
            ss << "\n";
            addItemValueToSS(BMInfoDataType::AcceleratorType);
            addItemValueToSS(BMInfoDataType::NumRuns);
            addItemValueToSS(BMInfoDataType::DataType);
            addItemValueToSS(BMInfoDataType::DataSize);
            addItemValueToSS(BMInfoDataType::DeviceName);
            addItemValueToSS(BMInfoDataType::WorkDivInit);
            addItemValueToSS(BMInfoDataType::WorkDivCopy);
            addItemValueToSS(BMInfoDataType::WorkDivMult);
            addItemValueToSS(BMInfoDataType::WorkDivAdd);
            addItemValueToSS(BMInfoDataType::WorkDivTriad);
            addItemValueToSS(BMInfoDataType::WorkDivDot);
            addItemValueToSS(BMInfoDataType::WorkDivNStream);
            addItemValueToSS(BMInfoDataType::CopyTimeFromAccToHost);

            // if the item is a string with delimited values get the item then the value at the index
            auto getItemFromStrList = [this](BMInfoDataType item, int index) -> std::string
            {
                std::string const str = metaDataMap.at(item);
                if(index < 1)
                {
                    throw std::invalid_argument("Index must be 1 or greater.");
                }

                std::istringstream iss(str);
                std::string token;
                int current_index = 1; // Start at 1 for 1-based indexing

                // Using ", " as the delimiter, we handle the token extraction manually
                while(std::getline(iss, token, ','))
                {
                    // Remove any leading spaces that may be left by `getline`
                    size_t start = token.find_first_not_of(' ');
                    if(start != std::string::npos)
                    {
                        token = token.substr(start);
                    }

                    if(current_index == index)
                    {
                        return token;
                    }
                    ++current_index;
                }

                throw std::out_of_range("Index out of range");
            };

            // Prepare Table Display
            // Table column names
            ss << std::endl;
            ss << std::left << std::setw(15) << typeToTypeStr(BMInfoDataType::KernelNames) << " " << std::left
               << std::setw(15) << typeToTypeStr(BMInfoDataType::KernelBandwidths) << " " << std::left << std::setw(10)
               << typeToTypeStr(BMInfoDataType::KernelMinTimes) << " " << std::left << std::setw(10)
               << typeToTypeStr(BMInfoDataType::KernelMaxTimes) << " " << std::left << std::setw(10)
               << typeToTypeStr(BMInfoDataType::KernelAvgTimes) << " " << std::left << std::setw(6)
               << typeToTypeStr(BMInfoDataType::KernelDataUsageValues) << " ";
            ss << std::endl;
            auto const kernelNamesStr = metaDataMap.at(BMInfoDataType::KernelNames);
            auto numberOfKernels = std::count(kernelNamesStr.begin(), kernelNamesStr.end(), ',') + 1;

            // Table rows. Print test results for each kernel line by line
            for(auto i = 1; i <= numberOfKernels; i++)
            {
                // Print the row for the kernel i.
                ss << " " << std::left << std::setw(15) << getItemFromStrList(BMInfoDataType::KernelNames, i) << " ";
                ss << std::left << std::setw(15) << getItemFromStrList(BMInfoDataType::KernelBandwidths, i) << " ";
                ss << std::left << std::setw(8) << getItemFromStrList(BMInfoDataType::KernelMinTimes, i) << " ";
                ss << std::left << std::setw(8) << getItemFromStrList(BMInfoDataType::KernelMaxTimes, i) << " ";
                ss << std::left << std::setw(8) << getItemFromStrList(BMInfoDataType::KernelAvgTimes, i) << " ";
                ss << std::left << std::setw(6) << getItemFromStrList(BMInfoDataType::KernelDataUsageValues, i) << " "
                   << std::endl;
            }

            return ss.str();
        }
    };

} // namespace
