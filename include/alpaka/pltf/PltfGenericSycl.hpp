/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>

#include <CL/sycl.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace alpaka
{
    //#############################################################################
    //! The SYCL device manager.
    class PltfGenericSycl : public concepts::Implements<ConceptPltf, PltfGenericSycl>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST PltfGenericSycl() = delete;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL platform device count get trait specialization.
        template<typename TPltf>
        struct GetDevCount<TPltf, std::enable_if_t<std::is_base_of_v<PltfGenericSycl, TPltf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDevCount() -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
                static auto platform = cl::sycl::platform{typename TPltf::selector{}};
                static auto devices = platform.get_devices();
#pragma clang diagnostic pop
                return devices.size();
            }
        };

        //#############################################################################
        //! The SYCL platform device get trait specialization.
        template<typename TPltf>
        struct GetDevByIdx<TPltf, std::enable_if_t<std::is_base_of_v<PltfGenericSycl, TPltf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDevByIdx(std::size_t const & devIdx)
            {
                using namespace cl::sycl;

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto exception_handler = [](cl::sycl::exception_list exceptions)
                {
                    for(std::exception_ptr e : exceptions)
                    {
                        try
                        {
                            std::rethrow_exception(e);
                        }
                        catch(const cl::sycl::exception& err)
                        {
                            auto ss_err = std::stringstream{};
                            ss_err << "Caught asynchronous SYCL exception: "
                                << err.what()
                                << " (" << err.get_cl_code() << ")";
                            throw std::runtime_error(ss_err.str());
                        }
                    }
                };

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
                static auto pf = platform{typename TPltf::selector{}};
                static auto devices = pf.get_devices();
                static auto ctx = context{devices, exception_handler};
#pragma clang diagnostic pop
                
                if(devIdx >= devices.size())
                {
                    auto ss_err = std::stringstream{};
                    ss_err << "Unable to return device handle for device "
                           << devIdx << ". There are only "
                           << devices.size() << " SYCL devices!";
                    throw std::runtime_error(ss_err.str());
                }

                auto sycl_dev = devices.at(devIdx);

                // Log this device.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDeviceProperties(sycl_dev);
#elif ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout << __func__ << sycl_dev.get_info<info::device::name>() << '\n';
#endif
                return typename DevType<TPltf>::type{sycl_dev, ctx};
            }

        private:
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            //! Prints all the device properties to std::cout.
            static auto printDeviceProperties(const cl::sycl::device& device) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                constexpr auto KiB = std::size_t{1024};
                constexpr auto MiB = KiB * KiB;

                std::cout << "Device type: ";
                switch(device.get_info<cl::sycl::info::device::device_type>())
                {
                    case cl::sycl::info::device_type::cpu:
                        std::cout << "CPU";
                        break;

                    case cl::sycl::info::device_type::gpu:
                        std::cout << "GPU";
                        break;

                    case cl::sycl::info::device_type::accelerator:
                        std::cout << "Accelerator";
                        break;

                    case cl::sycl::info::device_type::custom:
                        std::cout << "Custom";
                        break;

                    case cl::sycl::info::device_type::automatic:
                        std::cout << "Automatic";
                        break;

                    case cl::sycl::info::device_type::host:
                        std::cout << "Host";
                        break;

                    // The SYCL spec forbids the return of device_type::all
                    // Including this here to prevent warnings because of
                    // missing cases
                    case cl::sycl::info::device_type::all:
                        std::cout << "All";
                        break;
                }
                std::cout << std::endl;

                std::cout << "Name: "
                    << device.get_info<cl::sycl::info::device::name>()
                    << std::endl;

                std::cout << "Vendor: "
                    << device.get_info<cl::sycl::info::device::vendor>()
                    << std::endl;

                std::cout << "Vendor ID:"
                    << device.get_info<cl::sycl::info::device::vendor_id>()
                    << std::endl;

                std::cout << "Driver version: "
                    << device.get_info<cl::sycl::info::device::driver_version>()
                    << std::endl;

                std::cout << "OpenCL profile: "
                    << device.get_info<cl::sycl::info::device::profile>()
                    << std::endl;

                std::cout << "SYCL version: "
                    << device.get_info<cl::sycl::info::device::version>()
                    << std::endl;

                std::cout << "OpenCL C version: "
                    << device.get_info<cl::sycl::info::device::opencl_c_version>()
                    << std::endl;

                std::cout << "Extensions: " << std::endl;
                const auto extensions = device.get_info<cl::sycl::info::device::extensions>();
                for(const auto& ext : extensions)
                    std::cout << "\t" << ext << std::endl;

                std::cout << "Available compute units: "
                    << device.get_info<cl::sycl::info::device::max_compute_units>()
                    << std::endl;

                std::cout << "Maximum work item dimensions: ";
                auto dims = device.get_info<cl::sycl::info::device::max_work_item_dimensions>();
                std::cout << dims << std::endl;

                std::cout << "Maximum number of work items per dimension: ";
                auto item_dims = device.get_info<cl::sycl::info::device::max_work_item_sizes>(); 
                std::cout << "(";
                for(auto i = 0u; i < dims - 1u; ++i)
                    std::cout << item_dims[static_cast<int>(i)] << ", ";
                std::cout << item_dims[static_cast<int>(dims - 1u)] << ")" << std::endl;

                std::cout << "Maximum number of work items per work group: "
                    << device.get_info<cl::sycl::info::device::max_work_group_size>()
                    << std::endl;

                std::cout << "Preferred native vector width (char): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_char>()
                    << std::endl;

                std::cout << "Native ISA vector width (char): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_char>()
                    << std::endl;

                std::cout << "Preferred native vector width (short): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_short>()
                    << std::endl;

                std::cout << "Native ISA vector width (short): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_short>()
                    << std::endl;

                std::cout << "Preferred native vector width (int): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_int>()
                    << std::endl;

                std::cout << "Native ISA vector width (int): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_int>()
                    << std::endl;

                std::cout << "Preferred native vector width (long): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_long>()
                    << std::endl;

                std::cout << "Native ISA vector width (long): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_long>()
                    << std::endl;

                std::cout << "Preferred native vector width (float): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_float>()
                    << std::endl;

                std::cout << "Native ISA vector width (float): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_float>()
                    << std::endl;

                std::cout << "Preferred native vector width (double): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_double>()
                    << std::endl;

                std::cout << "Native ISA vector width (double): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_double>()
                    << std::endl;

                std::cout << "Preferred native vector width (half): "
                    << device.get_info<cl::sycl::info::device::preferred_vector_width_half>()
                    << std::endl;

                std::cout << "Native ISA vector width (half): "
                    << device.get_info<cl::sycl::info::device::native_vector_width_half>()
                    << std::endl;

                std::cout << "Maximum clock frequency: "
                    << device.get_info<cl::sycl::info::device::max_clock_frequency>()
                    << " MHz" << std::endl;

                std::cout << "Address space size: "
                    << device.get_info<cl::sycl::info::device::address_bits>()
                    << std::endl;

                std::cout << "Maximum size of memory object allocation: "
                    << device.get_info<cl::sycl::info::device::max_mem_alloc_size>()
                    << std::endl;

                std::cout << "Image support: ";
                auto has_img_support = device.get_info<cl::sycl::info::device::image_support>();
                std::cout << (has_img_support ? "Yes" : "No") << std::endl;

                if(has_img_support)
                {
                    std::cout << "Maximum number of simultaneous image object reads per kernel: "
                        << device.get_info<cl::sycl::info::device::max_read_image_args>()
                        << std::endl;

                    std::cout << "Maximum number of simultaneous image writes per kernel: "
                        << device.get_info<cl::sycl::info::device::max_write_image_args>()
                        << std::endl;

                    std::cout << "Maximum 1D/2D image width: "
                        << device.get_info<cl::sycl::info::device::image2d_max_width>()
                        << " px" << std::endl;

                    std::cout << "Maximum 2D image height: "
                        << device.get_info<cl::sycl::info::device::image2d_max_height>()
                        << " px" << std::endl;

                    std::cout << "Maximum 3D image width: "
                        << device.get_info<cl::sycl::info::device::image3d_max_width>()
                        << " px" << std::endl;

                    std::cout << "Maximum 3D image height: "
                        << device.get_info<cl::sycl::info::device::image3d_max_height>()
                        << " px" << std::endl;

                    std::cout << "Maximum 3D image depth: "
                        << device.get_info<cl::sycl::info::device::image3d_max_depth>()
                        << " px" << std::endl;

                    std::cout << "Maximum number of 1D/2D images per image array: "
                        << device.get_info<cl::sycl::info::device::image_max_array_size>()
                        << std::endl;

                    std::cout << "Maximum number of samplers per kernel: "
                        << device.get_info<cl::sycl::info::device::max_samplers>()
                        << std::endl;
                }

                std::cout << "Maximum kernel argument size: "
                    << device.get_info<cl::sycl::info::device::max_parameter_size>()
                    << " byte" << std::endl;

                std::cout << "Memory base address alignment: "
                    << device.get_info<cl::sycl::info::device::mem_base_addr_align>()
                    << " bit" << std::endl;

                auto print_fp_config = [](const std::string& fp,
                                          const std::vector<cl::sycl::info::fp_config>& conf)
                {
                    auto find_and_print = [&](cl::sycl::info::fp_config val)
                    {
                        auto it = std::find(begin(conf), end(conf), val);
                        std::cout << (it == std::end(conf) ? "No" : "Yes") << std::endl;
                    };

                    std::cout << fp << " denorm support: ";
                    find_and_print(cl::sycl::info::fp_config::denorm);

                    std::cout << fp << " INF & quiet NaN support: ";
                    find_and_print(cl::sycl::info::fp_config::inf_nan);

                    std::cout << fp << " round to nearest even support: ";
                    find_and_print(cl::sycl::info::fp_config::round_to_nearest);

                    std::cout << fp << " round to zero support: ";
                    find_and_print(cl::sycl::info::fp_config::round_to_zero);

                    std::cout << fp << " round to infinity support: ";
                    find_and_print(cl::sycl::info::fp_config::round_to_inf);

                    std::cout << fp << " IEEE754-2008 FMA support: ";
                    find_and_print(cl::sycl::info::fp_config::fma);

                    std::cout << fp << " correctly rounded divide/sqrt support: ";
                    find_and_print(cl::sycl::info::fp_config::correctly_rounded_divide_sqrt);

                    std::cout << fp << " software-implemented floating point operations: ";
                    find_and_print(cl::sycl::info::fp_config::soft_float);
                };

                if(device.has_extension("cl_khr_fp16"))
                {
                    const auto fp16_conf =
                        device.get_info<cl::sycl::info::device::half_fp_config>();
                    print_fp_config("FP16", fp16_conf);
                }

                const auto fp32_conf =
                    device.get_info<cl::sycl::info::device::single_fp_config>();
                print_fp_config("FP32", fp32_conf);

                if(device.has_extension("cl_khr_fp64"))
                {
                    const auto fp64_conf =
                        device.get_info<cl::sycl::info::device::double_fp_config>();
                    print_fp_config("FP64", fp64_conf);
                }

                std::cout << "Global memory cache type: ";
                auto has_global_mem_cache = false;
                switch(device.get_info<cl::sycl::info::device::global_mem_cache_type>())
                {
                    case cl::sycl::info::global_mem_cache_type::none:
                        std::cout << "none";
                        break;

                    case cl::sycl::info::global_mem_cache_type::read_only:
                        std::cout << "read-only";
                        has_global_mem_cache = true;
                        break;

                    case cl::sycl::info::global_mem_cache_type::read_write:
                        std::cout << "read-write";
                        has_global_mem_cache = true;
                        break;
                }
                std::cout << std::endl;

                if(has_global_mem_cache)
                {
                    std::cout << "Global memory cache line size: "
                        << device.get_info<cl::sycl::info::device::global_mem_cache_line_size>()
                        << " bytes" << std::endl;

                    std::cout << "Global memory cache size: "
                        << device.get_info<cl::sycl::info::device::global_mem_cache_size>() / KiB
                        << " KiB" << std::endl;
                }

                std::cout << "Global memory size: "
                    << device.get_info<cl::sycl::info::device::global_mem_size>() / MiB
                    << " MiB" << std::endl;

                std::cout << "Maximum constant buffer allocation size: "
                    << device.get_info<cl::sycl::info::device::max_constant_buffer_size>() / KiB
                    << " KiB" << std::endl;

                std::cout << "Maximum constant arguments per kernel: "
                    << device.get_info<cl::sycl::info::device::max_constant_args>()
                    << std::endl;

                std::cout << "Local memory type: ";
                auto has_local_memory = false;
                switch(device.get_info<cl::sycl::info::device::local_mem_type>())
                {
                    case cl::sycl::info::local_mem_type::none:
                        std::cout << "none";
                        break;

                    case cl::sycl::info::local_mem_type::local:
                        std::cout << "local";
                        has_local_memory = true;
                        break;

                    case cl::sycl::info::local_mem_type::global:
                        std::cout << "global";
                        has_local_memory = true;
                        break;
                }
                std::cout << std::endl;

                if(has_local_memory)
                {
                    std::cout << "Local memory size: "
                        << device.get_info<cl::sycl::info::device::local_mem_size>() / KiB
                        << " KiB" << std::endl;
                }

                std::cout << "Error correction support: "
                    << (device.get_info<cl::sycl::info::device::error_correction_support>() ?
                            "Yes" : "No")
                    << std::endl;

                std::cout << "Unified memory support: "
                    << (device.get_info<cl::sycl::info::device::host_unified_memory>() ?
                            "Yes" : "No")
                    << std::endl;

                std::cout << "Device timer resolution: "
                    << device.get_info<cl::sycl::info::device::profiling_timer_resolution>()
                    << " ns" << std::endl;

                std::cout << "Byte order: "
                    << (device.get_info<cl::sycl::info::device::is_endian_little>() ?
                            "Little Endian" : "Big Endian")
                    << std::endl;

                std::cout << "Device compiler available: "
                    << (device.get_info<cl::sycl::info::device::is_compiler_available>() ?
                            "Yes" : "No")
                    << std::endl;

                std::cout << "Device linker available: "
                    << (device.get_info<cl::sycl::info::device::is_linker_available>() ?
                            "Yes" : "No")
                    << std::endl;

                std::cout << "Queue profiling supported: "
                    << (device.get_info<cl::sycl::info::device::queue_profiling>() ?
                            "Yes" : "No")
                    << std::endl;

                std::cout << "Built-in OpenCL kernels: " << std::endl;
                const auto builtins = device.get_info<cl::sycl::info::device::built_in_kernels>();
                for(const auto& b : builtins)
                    std::cout << "\t" << b << std::endl;

                std::cout << "printf() buffer size: "
                    << device.get_info<cl::sycl::info::device::printf_buffer_size>() / MiB
                    << " MiB" << std::endl;

                std::cout << "Maximum number of subdevices: "
                    << device.get_info<cl::sycl::info::device::partition_max_sub_devices>()
                    << std::endl;
            }
#endif
        };
    }
}

#endif
