"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Software versions to be tested.
"""

from typing import Dict, List, Union
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

ALPAKA_VERSIONS: Dict[str, List[Union[str, int, float]]] = {
    GCC: [11, 12, 13],
    CLANG: [14, 15, 16, 17, 18, 19],
    NVCC: [12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6],
    HIPCC: [6.0, 6.1, 6.2],
    ICPX: ["2025.0"],
    UBUNTU: ["22.04", "24.04"],
    CMAKE: ["3.25.3", "3.26.4", "3.27.9", "3.28.6", "3.29.8", "3.30.3"],
    CXX_STANDARD: ["20"],
    BUILD_TYPE: BUILD_TYPES,
    # use only TEST_COMPILE_ONLY, because TEST_RUNTIME will be set manually depend on some
    # conditions later
    JOB_EXECUTION_TYPE: [JOB_EXECUTION_COMPILE_ONLY],
    MDSPAN: [ON, OFF],
}
