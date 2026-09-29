#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install Clang script does not support Windows or MacOS"
fi

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

parse_compiler_version "$APCI_DEVICE_COMPILER"

if [[ "$compiler_name" == "nvcc" ]]; then
    : "${APCI_HOST_COMPILER?'The device compiler was detected. Therefore the host compiler needs to be set'}"
    parse_compiler_version "$APCI_HOST_COMPILER"
fi

script_msg "Install Clang"

# If HIP is enabled, the clang compiler is handled by the rocm.sh script
if [[ "$compiler_name" == "clang" && "$APCI_HIP" == 0 ]]; then
    if agc-manager -e "clang@${compiler_version}"; then
        echo_green "use preinstalled clang@${compiler_version}"

        # TODO: Because of a bug in clang OpenMP apt package, no clang version is preinstalled
        # in alpaka-group-container and I cannot test what is the correct way to set the
        # required environment variables
        exit_error "Using preinstalled Clang provide by the agc-manager is not implemented yet."
    else
        install_msg "Clang $compiler_version"

        ci_wget https://apt.llvm.org/llvm-snapshot.gpg.key /etc/apt/trusted.gpg.d/apt.llvm.org.asc

        echo_run add-apt-repository -y \
            "deb https://apt.llvm.org/noble/ llvm-toolchain-noble-${compiler_version} main"

        clang_apt_package_list=(
            "clang-${compiler_version}"
            "libomp-${compiler_version}-dev"
            "clang-tools-${compiler_version}"
            "libclang-rt-${compiler_version}-dev"
        )

        if [[ "${APCI_CLANG_TIDY}" == "ON" ]]; then
            clang_apt_package_list+=("clang-tidy-${compiler_version}")
        fi

        DEBIAN_FRONTEND=noninteractive retry_cmd apt update
        # clang-tools is required, that CMake can setup clang as CUDA compiler
        # libclang-rt is required for the sanitizer
        quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y "${clang_apt_package_list[@]}"

        export APCI_CXX_COMPILER="/usr/bin/clang++-${compiler_version}"
    fi

    echo_run "$APCI_CXX_COMPILER" --version

    store_variable APCI_CXX_COMPILER
else
    echo_green "Skipped install Clang because it is not required for the job."
fi
