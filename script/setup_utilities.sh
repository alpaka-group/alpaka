#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# serveral helper function and tools for the CI

: "${ALPAKA_CI_OS_NAME?'ALPAKA_CI_OS_NAME must be specified'}"

# display a message in green
echo_green() {
    # macOS does not support bash colors
    if [ "$ALPAKA_CI_OS_NAME" != "macOS" ]; then
        echo -e "\e[1;32m$1\e[0m"
    else
        echo "$1"
    fi
}

# display a message in yellow
echo_yellow() {
    if [ "$ALPAKA_CI_OS_NAME" != "macOS" ]; then
        echo -e "\e[1;33m$1\e[0m"
    else
        echo "$1"
    fi
}

# display a message in red
echo_red() {
    if [ "$ALPAKA_CI_OS_NAME" != "macOS" ]; then
        echo -e "\e[1;31m$1\e[0m"
    else
        echo "$1"
    fi
}
