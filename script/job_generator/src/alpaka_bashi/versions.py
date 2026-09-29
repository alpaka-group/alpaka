"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Software versions to be tested.
"""

from copy import deepcopy

import bashi
import packaging.version
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    ALPAKA_ACC_ONEAPI_CPU_ENABLE,
    ALPAKA_ACC_ONEAPI_GPU_ENABLE,
    CLANG,
    CLANG_CUDA,
    CMAKE,
    COMPILERS,
    CXX_STANDARD,
    GCC,
    HIPCC,
    ICPX,
    NVCC,
    OFF,
    ON,
    UBUNTU,
)
from bashi.version.dependencies.base_version_support import ClangBase
from bashi.version.dependencies.clang_cuda import CLANG_CUDA_MAX_CUDA_VERSION, ClangCudaSDKSupport
from bashi.version.dependencies.hipcc import HIPCC_CLANG_VERSION
from bashi.version.dependencies.nvcc import NVCC_CLANG_MAX_VERSION, NVCC_GCC_MAX_VERSION, NvccHostSupport

from alpaka_bashi.globals import BUILD_TYPE, BUILD_TYPES, HWLOC

ALPAKA_VERSIONS: dict[str, list[str | int | float]] = {
    GCC: [12, 13, 14, 15, 16],
    CLANG: [17, 18, 19, 20, 21, 22, 23],
    NVCC: [12.5, 12.6, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4],
    HIPCC: [6.3, 6.4, 7.0, 7.1, 7.2, 7.14, 10.0],
    ICPX: ["2025.1", "2025.2", "2025.3", "2026.0", "2026.1"],
    UBUNTU: ["24.04"],
    CMAKE: ["3.25.3", "3.28.6", "3.31.12", "4.0.7", "4.4.0"],
    CXX_STANDARD: ["20"],
    BUILD_TYPE: BUILD_TYPES,
    HWLOC: [ON, OFF],
}


def _get_clang_cuda_versions() -> list[str | int | float]:
    """Return a list of Clang-CUDA versions. If there is no CUDA version
    bashi.versions.CLANG_CUDA_MAX_CUDA_VERSION which supports a specific Clang-CUDA, don't it add to
    the list.

    Returns:
        List[Union[str, int, float]]: List of Clang-CUDA versions.
    """
    min_cuda_version = packaging.version.parse(str(min(ALPAKA_VERSIONS[NVCC])))
    min_clang_cuda_version = packaging.version.parse("0")
    for clang_cuda_sdk in sorted(get_alpaka_version_relation().get_clang_cuda_max_cuda_version()):
        if min_cuda_version <= clang_cuda_sdk.cuda:
            min_clang_cuda_version = clang_cuda_sdk.clang_cuda
            break
    return [ver for ver in ALPAKA_VERSIONS[CLANG] if packaging.version.parse(str(ver)) >= min_clang_cuda_version]


def get_used_compiler_versions() -> dict[str, list[str | int | float]]:
    """Return a dict of used compiler and it's versions.

    Returns:
        dict[str, list[str | int | float]]: The key is the compiler name and value contains all
        versions.
    """
    return {name: versions for name, versions in ALPAKA_VERSIONS.items() if name in COMPILERS} | {
        CLANG_CUDA: _get_clang_cuda_versions()
    }


def get_software_versions_for_alpaka() -> dict[str, list[str | int | float]]:
    """Return dict of all compiler and software versions, which should be used as input for the
    combination generator.

    Raises:
        RuntimeError: If no valid Clang-CUDA versions exist.

    Returns:
        Dict[str, List[Union[str, int, float]]]: List of compiler and software versions.
    """

    clang_cuda_versions = _get_clang_cuda_versions()
    # The alpaka filter function cannot handle the case, that Clang-CUDA compiler are missing.
    # In the case, that the parameter-value-matrix is missing Clang-CUDA, we get a meaning full
    # error.
    if len(clang_cuda_versions) == 0:
        raise RuntimeError("Alpaka custom filter does not work without Clang-CUDA version.")

    return deepcopy(ALPAKA_VERSIONS) | {CLANG_CUDA: clang_cuda_versions}


def get_used_backends() -> list[str]:
    """Return the list of backends, used by alpaka."""
    return [
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_ONEAPI_CPU_ENABLE,
        ALPAKA_ACC_ONEAPI_GPU_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
    ]


def get_allowed_backend_combinations() -> list[bashi.CompilerBackendCombination]:
    """Return list of enabled backends for different host and device compiler combinations."""
    return [
        bashi.CompilerBackendCombination(
            GCC, GCC, [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]
        ),
        bashi.CompilerBackendCombination(
            GCC, GCC, [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE]
        ),
        bashi.CompilerBackendCombination(
            CLANG, CLANG, [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]
        ),
        bashi.CompilerBackendCombination(
            CLANG, CLANG, [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE]
        ),
        bashi.CompilerBackendCombination(GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE]),
        bashi.CompilerBackendCombination(CLANG, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE]),
        bashi.CompilerBackendCombination(CLANG_CUDA, CLANG_CUDA, [ALPAKA_ACC_GPU_CUDA_ENABLE]),
        bashi.CompilerBackendCombination(HIPCC, HIPCC, [ALPAKA_ACC_GPU_HIP_ENABLE]),
        bashi.CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_ONEAPI_CPU_ENABLE]),
        bashi.CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_ONEAPI_GPU_ENABLE]),
        bashi.CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE]),
    ]


def get_alpaka_version_relation() -> bashi.VersionRelation:
    """Returns:
    bashi.VersionRelation: bashi.VersionRelation object with alpaka specific modifications.
    """
    # bashi already offers numerous software relations. You can find all predefined relations
    # here: https://github.com/alpaka-group/bashi/blob/main/src/bashi/version/relation.py
    #
    # Relationships can be easily extended. The following example assumes that bashi has already
    # defined the relationship for Clang-CUDA 7 up to 17 and the CUDA SDK. The relationship is to be
    # extended up to Clang-CUDA 22.
    #
    # clang_cuda_max_cuda_version = CLANG_CUDA_MAX_CUDA_VERSION + [
    #    ClangCudaSDKSupport("18", "12.3"),
    #    ClangCudaSDKSupport("22", "13.0"),
    # ]
    #
    # bashi.VersionRelation(clang_cuda_max_cuda_version=clang_cuda_max_cuda_version)

    clang_cuda_max_cuda_version = CLANG_CUDA_MAX_CUDA_VERSION + [
        # Clang 20 + CUDA 12.8 is not official supported but alpaka 3.x is working with it
        ClangCudaSDKSupport("20", "12.8"),
        ClangCudaSDKSupport("22", "13.0"),
        ClangCudaSDKSupport("23", "13.2"),
    ]

    # Supported compilers:
    # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy
    nvcc_gcc_max_version = NVCC_GCC_MAX_VERSION + [
        NvccHostSupport("13.4", "16"),
    ]

    nvcc_clang_max_version = NVCC_CLANG_MAX_VERSION + [
        NvccHostSupport("13.4", "22"),
    ]

    hipcc_clang_version = HIPCC_CLANG_VERSION + [
        ClangBase("7.14", "23"),
        ClangBase("10.0", "23"),
    ]

    return bashi.VersionRelation(
        clang_cuda_max_cuda_version=clang_cuda_max_cuda_version,
        hipcc_clang_version=hipcc_clang_version,
        nvcc_gcc_max_version=nvcc_gcc_max_version,
        nvcc_clang_max_version=nvcc_clang_max_version,
    )
