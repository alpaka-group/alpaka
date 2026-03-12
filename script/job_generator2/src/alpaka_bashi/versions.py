"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Software versions to be tested.
"""

from typing import Dict, List, Union
import packaging.version
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

ALPAKA_VERSIONS: Dict[str, List[Union[str, int, float]]] = {
    GCC: [11, 12, 13],
    CLANG: [15, 16, 17, 18, 19],
    NVCC: [12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6],
    HIPCC: [6.0, 6.1, 6.2],
    ICPX: ["2025.0"],
    UBUNTU: ["22.04", "24.04"],
    CMAKE: ["3.25.3", "3.26.4", "3.27.9", "3.28.6", "3.29.8", "3.30.3"],
    CXX_STANDARD: ["20"],
    BUILD_TYPE: BUILD_TYPES,
    MDSPAN: [ON, OFF],
}


def _get_clang_cuda_versions() -> List[Union[str, int, float]]:
    """Return a list of Clang-CUDA versions. If there is no CUDA version
    bashi.versions.CLANG_CUDA_MAX_CUDA_VERSION which supports a specific Clang-CUDA, don't it add to
    the list.

    Returns:
        List[Union[str, int, float]]: List of Clang-CUDA versions.
    """
    min_cuda_version = packaging.version.parse(str(min(ALPAKA_VERSIONS[NVCC])))
    min_clang_cuda_version = packaging.version.parse("0")
    for clang_cuda_sdk in sorted(bashi.versions.CLANG_CUDA_MAX_CUDA_VERSION):
        if min_cuda_version <= clang_cuda_sdk.cuda:
            min_clang_cuda_version = clang_cuda_sdk.clang_cuda
            break
    return [
        ver
        for ver in ALPAKA_VERSIONS[CLANG]
        if packaging.version.parse(str(ver)) >= min_clang_cuda_version
    ]


def get_alpaka_version() -> Dict[str, List[Union[str, int, float]]]:
    """Return dict of all compiler and software versions, which should be used as input for the
    combination generator.

    Raises:
        RuntimeError: If no valid Clang-CUDA versions exist.

    Returns:
        Dict[str, List[Union[str, int, float]]]: List of compiler and software versions.
    """
    alpaka_version = ALPAKA_VERSIONS.copy()

    clang_cuda_versions = _get_clang_cuda_versions()
    if len(clang_cuda_versions) == 0:
        raise RuntimeError("Alpaka custom filter does not work without Clang-CUDA version.")
    alpaka_version[CLANG_CUDA] = clang_cuda_versions

    return alpaka_version
