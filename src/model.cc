/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>

namespace fasttext {

Model::Model(std::shared_ptr<Matrix> wi,
             std::shared_ptr<Matrix> wo,
             std::shared_ptr<Args> args,
             int32_t seed)
  : hidden_(args->dim), output_(wo->m_),
  grad_(args->dim), rng(seed), quant_(false)
{
  wi_ = wi;     // 输入, 初始为 [-1/dim, 1/dim] 的均匀分布
  wo_ = wo;     // 输出，词向量矩阵, init zero
  args_ = args;
  osz_ = wo->m_;
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  initSigmoid();
  initLog();
}

Model::~Model() {
  delete[] t_sigmoid;
  delete[] t_log;
}

void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi,
                               std::shared_ptr<QMatrix> qwo, bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

// LR 模型中的梯度计算公式为 grad = x*error, 参数 w 更新为 w = w + alpha * grad
// 代码中 wo_[target] 对应 LR 中的 w 参数, label-score 对应 error， 
// alpha 对应 alpha * error
real Model::binaryLogistic(int32_t target, bool label, real lr) {
  // wo_ 的 target 行点乘 hidden_ 向量
  // socre = P(Y=1|x) = 1 / (1 + e^(-x))
  // wo_ 与 hidden_ 的点乘是根据词向量论文中提到的 s() 函数, hidden_ 是输入
  // 词向量的均值
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  // 更新向量 grad_ += wo_[target] * alpha
  // Loss 对于 hidden_ 的梯度累加到 grad_ 上, 梯度公式是上面提到的 x*error, 
  // 此时把 wo_[target]看作是 x
  grad_.addRow(*wo_, target, alpha);
  // 更新 wo_[target] += hidden_ * alpha
  // Loss 对于 LR 参数的梯度累加到 wo_ 的对应行上
  wo_->addRow(hidden_, target, alpha);
  // 返回损失值 loss
  // 使用标准对数损失函数: -log(P(Y|X))
  if (label) {
    // 正样本
    return -log(score);
  } else {
    // 负样本
    // P(Y=0|x) = 1 - P(Y=1|x) = 1 - score
    return -log(1.0 - score);
  }
}

// 负采样方法
real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
    } else {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}

// 层次 softmax
real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

// 计算 output = softmax( wo_ * hidden )
void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    // output = wo_ * hidden
    output.mul(*wo_, hidden);
  }
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

// 计算 output_ = softmax( wo_ * hidden_ )
void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}


// ref: http://blog.csdn.net/l691899397/article/details/52291909
// wo_ 相当于权重矩阵, hiden_ 相当于输入, output_ 为 softmax 层的输出
// softmax 的偏导求解: 1. f(z) - 1 (正确预测的位置)  2. f(z)
real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    // grad_ += wo_[i] * alpha
    // 更新梯度
    // 对参数 hidden_ 求偏导为 wo_，反向传播到 wi_
    grad_.addRow(*wo_, i, alpha);
    // wo_[i] += hidden_ * alpha
    // 反向传播到 wo_, output_ = wo_ * hidden_, 对参数 wo_ 求偏导为 hidden_
    // 更新 softmax 参数
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]); // 标准对数损失 -log(P(Y|X))
}

// 计算隐藏层 hidden_ 向量, 对输入取平均
void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if(quant_) {
      hidden.addRow(*qwi_, *it);
    } else {
      // 将 wi_ 矩阵的第 *it 行加到 hidden_ 上
      hidden.addRow(*wi_, *it);
    }
  }
  // 求和后除以输入词个数，得到均值向量
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs) {
    dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, heap, hidden, output);
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap) {
  predict(input, k, heap, hidden_, output_);
}

void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap,
                      Vector& hidden, Vector& output) const {
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (heap.size() == k && log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Model::dfs(int32_t k, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f;
  if (quant_ && args_->qout) {
    f= sigmoid(qwo_->dotRow(hidden, node - osz_));
  } else {
    f= sigmoid(wo_->dotRow(hidden, node - osz_));
  }

  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
  dfs(k, tree[node].right, score + log(f), heap, hidden);
}

// target 是待预测的词
void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  // 计算前向传播：输入层 -> 隐层
  // wi_ ==> hidden_, 反向传播时对 hidden_ 求偏导
  // hidden_ 向量保存输入词向量的均值
  computeHidden(input, hidden_);
  // 根据输出层的不同结构，调用不同的函数，在各个函数中，
  // 不仅通过前向传播算出了 loss_，还进行了反向传播，计算出了 grad_
  // 更新了 wo_, grad_
  if (args_->loss == loss_name::ns) {
  // 1. 负采样
    loss_ += negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
  // 2. 层次 softmax
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
  // 3. 普通 softmax
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  // 如果是在训练分类器，就将 grad_ 除以 input_ 的大小
  // 原因不明
  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  // 反向传播，将 hidden_ 上的梯度传播到 wi_ 上的对应行
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }
}

// counts 为词出现词数，按降序排序
void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

// 初始化负采样点
void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

// 获得负采样例
int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

// counts 数组保存每个叶子节点的词频，降序排列
// 算法首先对输入的叶子节点进行一次排序, 时间复杂度 O(nlogn)，
// 然后确定两个下标 leaf 和 node，leaf 总是指向当前最小的叶子节点，
// node 总是指向当前最小的非叶子节点，
// 所以，最小的两个节点可以从 leaf, leaf - 1, node, node + 1 四个位置中取得，
// 时间复杂度 O(1)，每个非叶子节点都进行一次，所以总复杂度为 O(n)，
// 算法整体复杂度为 O(nlogn)
void Model::buildTree(const std::vector<int64_t>& counts) {
  // 分配所有节点的空间, osz 个叶子节点, osz-1 个非叶子节点
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  // leaf 指向当前未处理的叶子节点的最后一个，也就是权值最小的叶子节点
  int32_t leaf = osz_ - 1;
  // node 指向当前未处理的非叶子节点的第一个，也是权值最小的非叶子节点
  int32_t node = osz_;
  // 逐个构造所有非叶子节点（i >= osz_, i < 2 * osz - 1)
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];    // 最小的两个节点的下标
    // 计算权值最小的两个节点，候选只可能是 leaf, leaf - 1, node, node + 1
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    // 更新非叶子节点的属性
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }

  // 计算霍夫曼编码
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;  // 实现时是 tree[j].parent - osz_
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

// S(t) = 1 / (1 + e^(-t))
// ref: https://zh.wikipedia.org/wiki/S%E5%87%BD%E6%95%B0
void Model::initSigmoid() {
  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
  }
}

void Model::initLog() {
  t_log = new real[LOG_TABLE_SIZE + 1];
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log[i] = std::log(x);
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int i = int(x * LOG_TABLE_SIZE);
  return t_log[i];
}

// S(t) = 1 / (1 + e^(-t))
// ref: https://zh.wikipedia.org/wiki/S%E5%87%BD%E6%95%B0
real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

}
