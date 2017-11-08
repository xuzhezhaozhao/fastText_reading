#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

./fasttext supervised -input "${DATADIR}/gbar.train" -output "${RESULTDIR}/gbar" -dim 50 -lr 0.2 -wordNgrams 3 -minCount 1 -bucket 10000000 -epoch 25 -thread 4 -loss hs

./fasttext test "${RESULTDIR}/gbar.bin" "${DATADIR}/gbar.test"

./fasttext predict "${RESULTDIR}/gbar.bin" "${DATADIR}/gbar.test" > "${RESULTDIR}/gbar.test.predict"
