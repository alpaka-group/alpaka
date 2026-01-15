/* Copyright 2022 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/core/Config.hpp"
#include "alpaka/core/PP.hpp"

#include <type_traits>

// OpenMP 5.0+ support memory orders seq_cst, acq_rel, release, acquire, relaxed
#if ALPAKA_OMP >= ALPAKA_VERSION_NUMBER(2018, 11, 0)

#    define ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, ...)                                                           \
        do                                                                                                            \
        {                                                                                                             \
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)                                               \
            {                                                                                                         \
                _Pragma("omp atomic capture relaxed") __VA_ARGS__                                                     \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)                                          \
            {                                                                                                         \
                _Pragma("omp atomic capture acquire") __VA_ARGS__                                                     \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)                                          \
            {                                                                                                         \
                _Pragma("omp atomic capture release") __VA_ARGS__                                                     \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)                                           \
            {                                                                                                         \
                _Pragma("omp atomic capture acq_rel") __VA_ARGS__                                                     \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::SeqCst>)                                           \
            {                                                                                                         \
                _Pragma("omp atomic capture seq_cst") __VA_ARGS__                                                     \
            }                                                                                                         \
        } while(0)

// OpenMP 4.0+ supports seq_cst and relaxed (default)
#elif ALPAKA_OMP >= ALPAKA_VERSION_NUMBER(2013, 7, 0)

#    define ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, ...)                                                           \
        do                                                                                                            \
        {                                                                                                             \
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)                                               \
            {                                                                                                         \
                _Pragma("omp atomic capture") __VA_ARGS__                                                             \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                /** we fall back to a stronger atomic guarantee which may be slow */                                  \
                _Pragma("omp atomic capture seq_cst") __VA_ARGS__                                                     \
            }                                                                                                         \
        } while(0)

#else
// OpenMP 3.1 and below have a limited memory model and we emulate stronger atomics with flush. This may be slow
#    define ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, ...)                                                           \
        do                                                                                                            \
        {                                                                                                             \
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)                                               \
            {                                                                                                         \
                _Pragma("omp atomic capture") __VA_ARGS__                                                             \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                _Pragma("omp flush") _Pragma("omp atomic capture") __VA_ARGS__ _Pragma("omp flush")                   \
            }                                                                                                         \
        } while(0)
#endif

// Capture compare requires OpenMP 5.1
#if ALPAKA_OMP >= ALPAKA_VERSION_NUMBER(2020, 11, 0)
#    define ALPAKA_OMP_ATOMIC_CAPTURE_COMPARE_ORDER(TMemOrder, ...)                                                   \
        do                                                                                                            \
        {                                                                                                             \
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)                                               \
            {                                                                                                         \
                _Pragma("omp atomic capture compare relaxed") __VA_ARGS__                                             \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)                                          \
            {                                                                                                         \
                _Pragma("omp atomic capture compare acquire") __VA_ARGS__                                             \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)                                          \
            {                                                                                                         \
                _Pragma("omp atomic capture compare release") __VA_ARGS__                                             \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)                                           \
            {                                                                                                         \
                _Pragma("omp atomic capture compare acq_rel") __VA_ARGS__                                             \
            }                                                                                                         \
            else if constexpr(std::is_same_v<TMemOrder, mem_order::SeqCst>)                                           \
            {                                                                                                         \
                _Pragma("omp atomic capture compare seq_cst") __VA_ARGS__                                             \
            }                                                                                                         \
        } while(0)
#endif
