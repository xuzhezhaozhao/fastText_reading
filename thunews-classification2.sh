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
DATADIR=data/THUCNews

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

./fasttext supervised -input "${DATADIR}/thucnews.2.train" -output "${RESULTDIR}/thucnews-c.2" -dim 100 -lr 0.1 -ws 5 -neg 5 -loss ns -wordNgrams 2 -minCount 5 -bucket 10000000 -t 1e-4 -lrUpdateRate 100 -epoch 10 -thread 8

#./fasttext test "${RESULTDIR}/thucnews-c.bin" "${DATADIR}/thucnews.test"

#./fasttext predict "${RESULTDIR}/thucnews-c.bin" "${DATADIR}/thucnews.test" > "${RESULTDIR}/thucnews.test.predict"
