#!/bin/bash
#
# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Matthias Werner <Matthias.Werner1@tu-dresden.de>
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
# SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

source ./script/travis_retry.sh

source ./script/set.sh

cd docs/doxygen/html

git config --global user.email "action@github.com"
git config --global user.name "GitHub Action"

git add -f .

git log -n 3

git diff --quiet && git diff --staged --quiet || (git commit -m "Update documentation skip-checks: true"; git push origin gh-pages)

cd ../../../
