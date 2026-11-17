/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#define ALPAKA_PP_CAT(left, right) left##right
#define ALPAKA_PP_REMOVE_FIRST_COMMA_DO(ignore, ...) __VA_ARGS__
#define ALPAKA_PP_REMOVE_FIRST_COMMA(...) ALPAKA_PP_REMOVE_FIRST_COMMA_DO(0 __VA_ARGS__)

/** solution from https://stackoverflow.com/a/62984543
 * @{
 */
#define ALPAKA_PP_REMOVE_BRACKETS_DO(X) ALPAKAESC(ISHALPAKA X)
#define ISHALPAKA(...) ISHALPAKA __VA_ARGS__
#define ALPAKAESC(...) ALPAKAESC2(__VA_ARGS__)
#define ALPAKAESC2(...) VAN##__VA_ARGS__
#define VANISHALPAKA
/** @} */

#define ALPAKA_PP_REMOVE_BRACKETS(x) ALPAKA_PP_REMOVE_BRACKETS_DO(x)

// clang-format off
/* version number encoding
 * 4 digits for major version (max 9999)
 * 3 digits for minor version (max 999)
 * 5 digits for patch version (max 99999)
 * example: version 1.2.3 -> 0001 002 00003
 */
#define ALPAKA_VERSION_NUMBER(major, minor, patch)                                                                    \
    ((((major) % 10000llu) * 100000000llu) + (((minor) % 1000llu) * 100000llu) + ((patch) % 100000llu))

#define ALPAKA_VERSION_NUMBER_NOT_AVAILABLE ALPAKA_VERSION_NUMBER(0llu, 0llu, 0llu)
#define ALPAKA_VERSION_NUMBER_UNKNOWN ALPAKA_VERSION_NUMBER(9999llu, 999llu, 99999llu)

// version number conversion from vendor format to ALPAKA_VERSION_NUMBER
#define ALPAKA_YYYYMMDD_TO_VERSION(V) ALPAKA_VERSION_NUMBER(((V) / 10000llu), ((V) / 100llu) % 100llu, (V) % 100llu)

#define ALPAKA_YYYYMM_TO_VERSION(V) ALPAKA_VERSION_NUMBER(((V) / 100llu) % 10000llu, (V) % 100llu, 0llu)

#define ALPAKA_VVRRP_TO_VERSION(V)                                                                                    \
    ALPAKA_VERSION_NUMBER(((V) / 1000llu) % 10000llu, ((V) / 10llu) % 100llu, (V) % 10llu)

#define ALPAKA_VRP_TO_VERSION(V) ALPAKA_VERSION_NUMBER(((V) / 100llu) % 10000llu, ((V) / 10llu) % 10llu, (V) % 10llu)

#define ALPAKA_VRRPP_TO_VERSION(V)                                                                                    \
    ALPAKA_VERSION_NUMBER(((V) / 10000llu) % 10000llu, ((V) / 100llu) % 100llu, (V) % 100llu)
// clang-format on
