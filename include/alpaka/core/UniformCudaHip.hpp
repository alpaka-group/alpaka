/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Andrea Bocci, Bernhard
 * Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#if !defined(ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE)
#    error This is an internal header file, and should never be included directly.
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#error This file should not be included with ALPAKA_ACC_GPU_CUDA_ENABLED and ALPAKA_ACC_GPU_HIP_ENABLED both defined.
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(alpaka_core_UniformCudaHip_hpp_CUDA)                             \
    || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(alpaka_core_UniformCudaHip_hpp_HIP)

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(alpaka_core_UniformCudaHip_hpp_CUDA)
#        define alpaka_core_UniformCudaHip_hpp_CUDA
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(alpaka_core_UniformCudaHip_hpp_HIP)
#        define alpaka_core_UniformCudaHip_hpp_HIP
#    endif


#    include <alpaka/core/BoostPredef.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <array>
#    include <stdexcept>
#    include <string>
#    include <tuple>
#    include <type_traits>

namespace alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail
{
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using Error_t = cudaError;
#    else
    using Error_t = hipError_t;
#    endif

    //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
    template<bool TThrow>
    ALPAKA_FN_HOST inline void rtCheck(
        Error_t const& error,
        char const* desc,
        char const* file,
        int const& line) noexcept(!TThrow)
    {
        if(error != ALPAKA_API_PREFIX(Success))
        {
            auto const sError = std::string{
                std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '"
                + ALPAKA_API_PREFIX(GetErrorName)(error) + "': '"
                + std::string(ALPAKA_API_PREFIX(GetErrorString)(error)) + "'!"};

            if constexpr(!TThrow || ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
                std::cerr << sError << std::endl;

            ALPAKA_DEBUG_BREAK;
            // reset the last error to allow user side error handling. Using std::ignore to discard unneeded
            // return values is suggested by the C++ core guidelines.
            std::ignore = ALPAKA_API_PREFIX(GetLastError)();

            if constexpr(TThrow)
                throw std::runtime_error(sError);
        }
    }

    //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
    // NOTE: All ignored errors have to be convertible to Error_t.
    template<bool TThrow, typename... TErrors>
    ALPAKA_FN_HOST inline void rtCheckIgnore(
        Error_t const& error,
        char const* cmd,
        char const* file,
        int const& line,
        TErrors&&... ignoredErrorCodes) noexcept(!TThrow)
    {
        if(error != ALPAKA_API_PREFIX(Success))
        {
            std::array<Error_t, sizeof...(ignoredErrorCodes)> const aIgnoredErrorCodes{ignoredErrorCodes...};

            // If the error code is not one of the ignored ones.
            if(std::find(std::cbegin(aIgnoredErrorCodes), std::cend(aIgnoredErrorCodes), error)
               == std::cend(aIgnoredErrorCodes))
            {
                rtCheck<TThrow>(error, ("'" + std::string(cmd) + "' returned error ").c_str(), file, line);
            }
            else
            {
                // reset the last error to avoid propagation to the next CUDA/HIP API call. Using std::ignore
                // to discard unneeded return values is recommended by the C++ core guidelines.
                std::ignore = ALPAKA_API_PREFIX(GetLastError)();
            }
        }
    }

    //! CUDA/HIP runtime API last error checking with log and exception.
    template<bool TThrow>
    ALPAKA_FN_HOST inline void rtCheckLastError(char const* desc, char const* file, int const& line) noexcept(!TThrow)
    {
        Error_t const error(ALPAKA_API_PREFIX(GetLastError)());
        rtCheck<TThrow>(error, desc, file, line);
    }
} // namespace alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                     \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckLastError<true>(                           \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckIgnore<true>(                              \
                cmd,                                                                                                  \
                #cmd,                                                                                                 \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                __VA_ARGS__)
#    else
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif
//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                     \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckLastError<true>(                           \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckIgnore<true>(                              \
                cmd,                                                                                                  \
                #cmd,                                                                                                 \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                ##__VA_ARGS__)
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

//! CUDA/HIP runtime error checking with log and exception.
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd)

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
//! CUDA/HIP runtime error checking with log, ignoring specific error values
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                            \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckLastError<false>(                          \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckIgnore<false>(                             \
                cmd,                                                                                                  \
                #cmd,                                                                                                 \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                __VA_ARGS__)
#    else
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif
//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                            \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckLastError<false>(                          \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE::detail::rtCheckIgnore<false>(                             \
                cmd,                                                                                                  \
                #cmd,                                                                                                 \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                ##__VA_ARGS__)
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

//! CUDA/HIP runtime error checking with log.
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd)

//! Macros explicitly for the CUDA version
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#        if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
//! CUDA runtime error checking with log and exception, ignoring specific error values
#            define ALPAKA_CUDA_RT_CHECK_IGNORE(cmd, ...)                                                             \
                ::alpaka::cuda::detail::rtCheckLastError<true>(                                                       \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::cuda::detail::rtCheckIgnore<true>(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#        else
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#            endif
//! CUDA runtime error checking with log and exception, ignoring specific error values
#            define ALPAKA_CUDA_RT_CHECK_IGNORE(cmd, ...)                                                             \
                ::alpaka::cuda::detail::rtCheckLastError<true>(                                                       \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::cuda::detail::rtCheckIgnore<true>(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic pop
#            endif
#        endif

//! CUDA runtime error checking with log and exception.
#        define ALPAKA_CUDA_RT_CHECK(cmd) ALPAKA_CUDA_RT_CHECK_IGNORE(cmd)

#        if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
//! CUDA runtime error checking with log, ignoring specific error values
#            define ALPAKA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                                     \
                ::alpaka::cuda::detail::rtCheckLastError<false>(                                                      \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::cuda::detail::rtCheckIgnore<false>(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#        else
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#            endif
//! CUDA runtime error checking with log and exception, ignoring specific error values
#            define ALPAKA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                                     \
                ::alpaka::cuda::detail::rtCheckLastError<false>(                                                      \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::cuda::detail::rtCheckIgnore<false>(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic pop
#            endif
#        endif

//! CUDA runtime error checking with log.
#        define ALPAKA_HIP_RT_CHECK_NOEXCEPT(cmd) ALPAKA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd)

#    endif // ALPAKA_ACC_GPU_CUDA_ENABLED

//! Macros explicitly for the HIP version
#    ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#        if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
//! HIP runtime error checking with log and exception, ignoring specific error values
#            define ALPAKA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                              \
                ::alpaka::hip::detail::rtCheckLastError<true>(                                                        \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::hip::detail::rtCheckIgnore<true>(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#        else
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#            endif
//! HIP runtime error checking with log and exception, ignoring specific error values
#            define ALPAKA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                              \
                ::alpaka::hip::detail::rtCheckLastError<true>(                                                        \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::hip::detail::rtCheckIgnore<true>(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic pop
#            endif
#        endif

//! HIP runtime error checking with log and exception.
#        define ALPAKA_HIP_RT_CHECK(cmd) ALPAKA_HIP_RT_CHECK_IGNORE(cmd)

#        if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
//! HIP runtime error checking with log, ignoring specific error values
#            define ALPAKA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                                     \
                ::alpaka::hip::detail::rtCheckLastError<false>(                                                       \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::hip::detail::rtCheckIgnore<false>(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#        else
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#            endif
//! HIP runtime error checking with log and exception, ignoring specific error values
#            define ALPAKA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                                     \
                ::alpaka::hip::detail::rtCheckLastError<false>(                                                       \
                    "'" #cmd "' A previous API call (not this one) set the error ",                                   \
                    __FILE__,                                                                                         \
                    __LINE__);                                                                                        \
                ::alpaka::hip::detail::rtCheckIgnore<false>(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#            if BOOST_COMP_CLANG
#                pragma clang diagnostic pop
#            endif
#        endif

//! HIP runtime error checking with log.
#        define ALPAKA_HIP_RT_CHECK_NOEXCEPT(cmd) ALPAKA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd)

#    endif // ALPAKA_ACC_GPU_HIP_ENABLED

#endif
