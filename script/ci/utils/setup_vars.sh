#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

set -eu

# setup environment variables depending on os environment (on local system, GitLab CI, GitHub Action ...)

# set the required memory per build thread in GB
export APCI_REQUIRED_RAM_PER_BUILD_THREAD_BYTES=$((2 * 1024 ** 3))

if [[ -n ${GITHUB_ACTIONS+x} ]]; then
    # force color output
    export TERM=xterm-256color
    export _APCI_FORCE_COLOR_OUTPUT=1

    export APCI_OS_NAME="$RUNNER_OS"

    if [[ -z ${APCI_EXEC_CPU_SERIAL+x} ]]; then
        export APCI_EXEC_CPU_SERIAL=OFF
    fi

    if [[ -z ${APCI_HWLOC+x} ]]; then
        export APCI_HWLOC=ON
    fi

    # The Github Actions are handwritten, therefore set some undefined variables to default
    # variables. GitLab CI will not do it, because all jobs are generated and default
    # values for undefined variables are an error source.
    if [[ -z ${APCI_HIP+x} ]]; then
        export APCI_HIP=0
    fi

    # there are no GPU runner on GitHub, therefore simply choose one architecture
    APCI_AMD_GPU_ARCH=gfx90a
    export APCI_AMD_GPU_ARCH

    if [[ ! "${APCI_DEVICE_COMPILER}" =~ "nvcc" ]] && [[ -z ${APCI_CUDA+x} ]]; then
        export APCI_CUDA=0
    fi

    # GitHub actions has no free GPU runner, therefore choose simply a single SM level
    export APCI_CUDA_SM_LEVEL=80

    if [[ ! "${APCI_DEVICE_COMPILER}" =~ "icpx" ]] && [[ -z ${APCI_ONEAPI+x} ]]; then
        export APCI_ONEAPI=0
    fi

    if [[ "${APCI_ONEAPI}" == 0 ]] && [[ -z ${APCI_ONEAPI_TARGET+x} ]]; then
        export APCI_ONEAPI_TARGET=none
    fi

    if [[ -z ${APCI_TBB+x} ]]; then
        export APCI_TBB="OFF"
    fi

    if [[ -z ${APCI_SANITIZER_ASAN+x} ]]; then
        export APCI_SANITIZER_ASAN="OFF"
    fi
    if [[ -z ${APCI_SANITIZER_TSAN+x} ]]; then
        export APCI_SANITIZER_TSAN="OFF"
    fi
    if [[ -z ${APCI_SANITIZER_LSAN+x} ]]; then
        export APCI_SANITIZER_LSAN="OFF"
    fi
    if [[ -z ${APCI_SANITIZER_UBSAN+x} ]]; then
        export APCI_SANITIZER_UBSAN="OFF"
    fi

    if [[ -z ${APCI_SIMD+x} ]]; then
        export APCI_SIMD=DEFAULT
    fi

    if [[ -z ${APCI_CLANG_TIDY+x} ]]; then
        export APCI_CLANG_TIDY=OFF
    fi

    if [[ -z ${APCI_OMP+x} ]]; then
        # if no other backend is enabled, use OpenMP
        if [[ "$APCI_HIP" == 0 && "$APCI_ONEAPI" == 0 && "$APCI_CUDA" == 0 && "$APCI_TBB" == "OFF" ]]; then
            APCI_OMP="ON"
        else
            APCI_OMP="OFF"
        fi
    fi
    export APCI_OMP
fi

if [[ -n ${GITLAB_CI+x} ]]; then
    # force color output
    export TERM=xterm-256color
    export _APCI_FORCE_COLOR_OUTPUT=1

    # workaround github clone issues: https://github.com/orgs/community/discussions/206581
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="http.https://github.com.version"
    export GIT_CONFIG_VALUE_0="HTTP/1.1"

    if echo "${CI_RUNNER_EXECUTABLE_ARCH}" | grep -q -i "linux"; then
        export APCI_OS_NAME=Linux
    fi
    # not validated because we didn't used a windows runner yet
    if echo "${CI_RUNNER_EXECUTABLE_ARCH}" | grep -q -i "windows"; then
        export APCI_OS_NAME=Windows
    fi
    # not validated because we didn't used a MacOS runner yet
    if echo "${CI_RUNNER_EXECUTABLE_ARCH}" | grep -q -i "macos"; then
        export APCI_OS_NAME=macOS
    fi

    export APCI_IMAGE_NAME="$CI_JOB_IMAGE"
    if [[ "${CI_COMMIT_REF_NAME}" =~ "pr-" ]]; then
        IFS='/' read -r _pr_number _repo_owner _repo APCI_BRANCH_NAME <<<"${CI_COMMIT_REF_NAME}"
        export APCI_GIT_URL="https://github.com/${_repo_owner}/${_repo}.git"
        export APCI_BRANCH_NAME
        unset _pr_number _repo_owner _repo
    else
        export APCI_GIT_URL="https://github.com/alpaka-group/alpaka.git"
        export APCI_BRANCH_NAME="${CI_COMMIT_REF_NAME}"
    fi

    if [[ "$APCI_HIP" != 0 ]]; then
        # on the GPU runner, the variable CI_GPU_ARCH is predefined
        if [[ -n ${CI_GPU_ARCH+x} ]]; then
            APCI_AMD_GPU_ARCH=$CI_GPU_ARCH
        else
            # in compile only jobs, use simply this architecture
            APCI_AMD_GPU_ARCH=gfx90a
        fi
    fi
    export APCI_AMD_GPU_ARCH

    if [[ "$APCI_CUDA" != 0 ]]; then
        # on the GPU runner, the variable CI_GPU_ARCH is predefined
        if [[ -n ${CI_GPU_ARCH+x} ]]; then
            APCI_CUDA_SM_LEVEL="${CI_GPU_ARCH}"
        else
            # in compile only jobs, use simply this architecture
            APCI_CUDA_SM_LEVEL=80
        fi
    fi
    export APCI_CUDA_SM_LEVEL
fi
