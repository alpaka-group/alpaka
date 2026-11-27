"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the variables of the GitLab CI test job yaml.
"""

from typing import Any

import bashi
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    ALPAKA_ACC_ONEAPI_CPU_ENABLE,
    ALPAKA_ACC_ONEAPI_GPU_ENABLE,
    CLANG_CUDA,
    CMAKE,
    DEVICE_COMPILER,
    HOST_COMPILER,
    NVCC,
    OFF_VER,
    ON_VER,
)
from typeguard import typechecked

from alpaka_bashi.globals import CI_PIPELINE_COMPILE_ONLY_VER, HWLOC, JOB_EXECUTION_TYPE


@typechecked
def set_variables(job_body: dict[str, Any], combination: bashi.Combination):
    """Set the variables of the GitLab CI test job yaml depending on the combination.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    if "variables" not in job_body:
        job_body["variables"] = {}

    set_generic_variables(job_body["variables"], combination)

    if combination[ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE].version == ON_VER:
        job_body["variables"]["APCI_OMP"] = "ON"

    if combination[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE].version == ON_VER:
        job_body["variables"]["APCI_TBB"] = "ON"

    if combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
        job_body["variables"]["APCI_CUDA"] = str(combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version)
        if combination[DEVICE_COMPILER].name == NVCC:
            job_body["variables"]["APCI_HOST_COMPILER"] = (
                f"{combination[HOST_COMPILER].name}@{str(combination[HOST_COMPILER].version)}"
            )

    if combination[ALPAKA_ACC_GPU_HIP_ENABLE].version == ON_VER:
        job_body["variables"]["APCI_HIP"] = str(combination[DEVICE_COMPILER].version)

    if ON_VER in (combination[ALPAKA_ACC_ONEAPI_CPU_ENABLE].version, combination[ALPAKA_ACC_ONEAPI_GPU_ENABLE].version):
        job_body["variables"]["APCI_ONEAPI"] = str(combination[DEVICE_COMPILER].version)
        if combination[ALPAKA_ACC_ONEAPI_CPU_ENABLE].version == ON_VER:
            job_body["variables"]["APCI_ONEAPI_TARGET"] = "cpu"
        if combination[ALPAKA_ACC_ONEAPI_GPU_ENABLE].version == ON_VER:
            job_body["variables"]["APCI_ONEAPI_TARGET"] = "intel_gpu"


@typechecked
def set_generic_variables(variables: dict[str, Any], combination: bashi.Combination):
    """Set variables, which are valid for all jobs. Disable dependencies and sanitizer by default.

    Args:
        variables (Dict[str, Any]): variable section of a GitLab CI job
        combination (bashi.Combination): combination
    """
    variables["APCI_ALPAKA_ROOT"] = "$CI_PROJECT_DIR"
    variables["APCI_ONEAPI_TARGET"] = "none"
    variables["APCI_SIMD"] = "DEFAULT"
    variables["APCI_CLANG_TIDY"] = "OFF"

    dependencies = ["APCI_OMP", "APCI_TBB"]
    gpu_dependencies = ["APCI_CUDA", "APCI_HIP", "APCI_ONEAPI"]
    sanitizers = ["APCI_SANITIZER_ASAN", "APCI_SANITIZER_TSAN", "APCI_SANITIZER_LSAN", "APCI_SANITIZER_UBSAN"]

    for var in dependencies + sanitizers:
        variables[var] = "OFF"

    for var in gpu_dependencies:
        variables[var] = 0

    if combination[DEVICE_COMPILER].name != CLANG_CUDA:
        variables["APCI_DEVICE_COMPILER"] = (
            f"{combination[DEVICE_COMPILER].name}@{str(combination[DEVICE_COMPILER].version)}"
        )
    else:
        variables["APCI_DEVICE_COMPILER"] = f"clang@{str(combination[DEVICE_COMPILER].version)}"

    variables["APCI_EXEC_CPU_SERIAL"] = bashi.on_off_ver_to_str(combination[ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE].version)
    variables["APCI_CMAKE"] = str(combination[CMAKE].version)
    variables["APCI_HWLOC"] = bashi.on_off_ver_to_str(combination[HWLOC].version)
    variables["APCI_RUN_CTEST"] = (
        "OFF" if combination[JOB_EXECUTION_TYPE].version == CI_PIPELINE_COMPILE_ONLY_VER else "ON"
    )

    # debug variable for HIP
    # variables["AMD_LOG_LEVEL"] = "5"
    # with 2 hardware queues the test evReEnqueueWithSomeoneWaitsForEventInOrderLifetimeRelease is failing
    variables["GPU_MAX_HW_QUEUES"] = "3"
    # variables["HSA_ENABLE_SDMA"] = "0"
