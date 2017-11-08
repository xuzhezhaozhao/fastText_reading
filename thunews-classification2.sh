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

./fasttext supervised -input "${DATADIR}/thucnews.2.train" -output "${RESULTDIR}/thucnews-c.2" -dim 50 -lr 0.1 -wordNgrams 3 -minCount 1 -bucket 10000000 -epoch 20 -thread 8

#./fasttext test "${RESULTDIR}/thucnews-c.bin" "${DATADIR}/thucnews.test"

#./fasttext predict "${RESULTDIR}/thucnews-c.bin" "${DATADIR}/thucnews.test" > "${RESULTDIR}/thucnews.test.predict"
