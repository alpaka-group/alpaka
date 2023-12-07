#!/bin/bash
#
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/travis_retry.sh

source ./script/set.sh

travis_retry sudo apt-get -y --quiet install --no-install-recommends doxygen graphviz
