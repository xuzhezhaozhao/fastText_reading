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

./fasttext skipgram -input "${DATADIR}"/thucnews.preprocessed.all -output "${RESULTDIR}"/thucnews-w -lr 0.025 -dim 100 \
  -ws 5 -epoch 1 -minCount 5 -neg 5 -loss ns -bucket 2000000 \
  -minn 3 -maxn 6 -thread 4 -t 1e-4 -lrUpdateRate 100
