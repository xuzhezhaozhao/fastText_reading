/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace fasttext {

enum class model_name : int {cbow=1, sg, sup};
enum class loss_name : int {hs=1, ns, softmax};

class Args {
  private:
    std::string lossToString(loss_name);

  public:
    Args();
    std::string input;
    std::string test;
    std::string output;
    double lr;      // learning rate
    int lrUpdateRate;   // change the rate of updates for the learning rate
    int dim;    // size of word vectors
    int ws;     // size of the context window
    int epoch;  // number of epochs
    int minCount;   // minimal number of word occurences
    int minCountLabel;  // minimal number of label occurences
    int neg;        // number of negatives sampled
    int wordNgrams; // max length of word ngram
    loss_name loss; // loss function {ns, hs, softmax} [softmax]
    model_name model;
    int bucket;     // number of buckets
    int minn;       // min length of char ngram
    int maxn;       // max length of char ngram
    int thread;
    double t;       // sampling threshold
    std::string label;  // labels prefix
    int verbose;    // verbosity level [2]
    std::string pretrainedVectors;  // pretrained word vectors for supervised learning
    int saveOutput; // whether output params should be saved

    // The following arguments for quantization are optional
    bool qout;      // quantizing the classifier
    bool retrain;   // finetune embeddings if a cutoff is applied
    bool qnorm;     // quantizing the norm separately
    size_t cutoff;  // number of words and ngrams to retain
    size_t dsub;    // size of each sub-vector

    void parseArgs(const std::vector<std::string>& args);
    void printHelp();
    void printBasicHelp();
    void printDictionaryHelp();
    void printTrainingHelp();
    void printQuantizationHelp();
    void save(std::ostream&);
    void load(std::istream&);
};

}

#endif
