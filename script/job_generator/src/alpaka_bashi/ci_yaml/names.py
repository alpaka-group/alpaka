"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate CI job names depending on the value-versions of a combination.
"""

import bashi
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_ONEAPI_CPU_ENABLE,
    ALPAKA_ACC_ONEAPI_GPU_ENABLE,
    CLANG_CUDA,
    CMAKE,
    CXX_STANDARD,
    DEVICE_COMPILER,
    HOST_COMPILER,
    NVCC,
    ON_VER,
    UBUNTU,
)
from typeguard import typechecked

from alpaka_bashi.globals import (
    BUILD_TYPE,
    CMAKE_DEBUG_VER,
    CMAKE_RELEASE_VER,
    HWLOC,
    JOB_EXECUTION_COMPILE_ONLY_VER,
    JOB_EXECUTION_TYPE,
)
from alpaka_bashi.versions import get_spack_alias


@typechecked
def get_job_suffix(combination: bashi.Combination) -> str:
    """Generate a suffix for a CI job name depending on the value-versions of a combination.

    Args:
        comb (Dict[str, Tuple[str, str]]): combination.

    Returns:
        str: name suffix.
    """
    version_str = ""
    enabled_software = {
        HWLOC: "_hwloc",
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: "_omp",
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: "_tbb",
        ALPAKA_ACC_ONEAPI_CPU_ENABLE: "_oneapi_cpu",
        ALPAKA_ACC_ONEAPI_GPU_ENABLE: "_oneapi_gpu",
    }

    for software in [
        CMAKE,
        UBUNTU,
        CXX_STANDARD,
        HWLOC,
        BUILD_TYPE,
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
        ALPAKA_ACC_ONEAPI_CPU_ENABLE,
        ALPAKA_ACC_ONEAPI_GPU_ENABLE,
    ]:
        if software in combination:
            if combination[software].name == CXX_STANDARD:
                version_str += f"_cxx{str(combination[software].version)}"
            elif combination[software].name == UBUNTU:
                version_str += (
                    f"_{combination[software].name}{bashi.ubuntu_version_to_string(combination[software].version)}"
                )
            elif combination[software].name == BUILD_TYPE:
                continue
            elif combination[software].name in enabled_software:
                if combination[software].version == ON_VER:
                    version_str += enabled_software[combination[software].name]
            else:
                version_str += f"_{combination[software].name}{str(combination[software].version)}"

    # make sure, that the build type is at the end of the name
    if combination[BUILD_TYPE].version == CMAKE_RELEASE_VER:
        version_str += "_release"
    if combination[BUILD_TYPE].version == CMAKE_DEBUG_VER:
        version_str += "_debug"

    return version_str


@typechecked
def get_job_name(comb: bashi.Combination) -> str:
    """Generate a CI job name depending on the value-versions of a combination.

    Args:
        comb (Dict[str, Tuple[str, str]]): combination.

    Returns:
        str: name suffix.
    """
    # the job name starts with the device compiler
    job_name = f"linux_{comb[DEVICE_COMPILER].name}{str(comb[DEVICE_COMPILER].version)}"
    # if the nvcc is the device compiler, add also the host compiler to the name
    if comb[DEVICE_COMPILER].name == NVCC:
        job_name = job_name + f"-{comb[HOST_COMPILER].name}{str(comb[HOST_COMPILER].version)}"
    # if Clang-CUDA is the device compiler, add also the CUDA SDK version to the name
    if comb[DEVICE_COMPILER].name == CLANG_CUDA:
        job_name = job_name + f"-cuda{str(comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version)}"

    if comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_COMPILE_ONLY_VER:
        job_name += "_compile_only"

    return job_name + get_job_suffix(comb)


@typechecked
def get_spack_build_job_name(
    ubuntu_version: bashi.ValueVersion, software_name: bashi.ValueName, software_version: bashi.ValueVersion
) -> str:
    """
    Return the name of a spack package build job.

    Args:
        ubuntu_version (bashi.ValueVersion): Ubuntu version of the base image.
        software_name (bashi.ValueName): Name of the software.
        software_version (bashi.ValueVersion): Software version

    Returns:
        str: Job name
    """
    return (
        f"spack_build_{bashi.ubuntu_version_to_string(ubuntu_version)}_"
        f"{get_spack_alias()[software_name].package_name}{software_version}"
    )
