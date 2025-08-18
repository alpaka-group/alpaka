"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Validate if a combination parameter-values is valid for alpaka.
"""

import sys
from bashi.globals import ON, OFF
import bashiValidate
import alpaka_bashi


def main():
    """Entry point function."""

    validator = bashiValidate.Validator()
    validator.add_custom_filter(alpaka_bashi.AlpakaFilter())
    validator.add_string_parameter(
        alpaka_bashi.BUILD_TYPE, "CMake build type.", alpaka_bashi.BUILD_TYPES_NAMES
    )
    validator.add_software_version_parameter(
        name=alpaka_bashi.MDSPAN, help_text="Build with C++23 std::mdspan.", choices=["ON", "OFF"]
    )
    validator.add_known_version(alpaka_bashi.MDSPAN, [OFF, ON])
    sys.exit(int(not validator.validate()))


if __name__ == "__main__":
    main()
