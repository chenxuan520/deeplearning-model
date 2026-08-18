#include "train/evaluator.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace az {

namespace {

bool CopyBatchNormState(deeplearning::BatchNorm2D &dst,
                        const deeplearning::BatchNorm2D &src) {
  if (dst.config().channels_ != src.config().channels_) {
    return false;
  }
  return dst.set_running_mean(src.running_mean()) ==
             deeplearning::BatchNorm2D::SUCCESS &&
         dst.set_running_variance(src.running_variance()) ==
             deeplearning::BatchNorm2D::SUCCESS;
}

} // namespace

// Copies trainable parameters AND batch-norm running statistics between two
// networks with identical configs. Returns false on any shape mismatch.
bool AssignWeights(deeplearning::PolicyValueResNet &dst,
                   deeplearning::PolicyValueResNet &src) {
  auto dst_params = dst.TrainableParameters();
  auto src_params = src.TrainableParameters();
  if (dst_params.size() != src_params.size()) {
    return false;
  }
  for (std::size_t i = 0; i < dst_params.size(); ++i) {
    if (dst_params[i].value_->size() != src_params[i].value_->size()) {
      return false;
    }
    *dst_params[i].value_ = *src_params[i].value_;
  }
  // Batch-norm running statistics are buffers, not trainable parameters:
  // copy them explicitly or inference would normalize with mean=0/var=1.
  if (!CopyBatchNormState(dst.stem_norm(), src.stem_norm()) ||
      !CopyBatchNormState(dst.policy_norm(), src.policy_norm()) ||
      !CopyBatchNormState(dst.value_norm(), src.value_norm())) {
    return false;
  }
  auto &dst_blocks = dst.blocks();
  auto &src_blocks = src.blocks();
  if (dst_blocks.size() != src_blocks.size()) {
    return false;
  }
  for (std::size_t i = 0; i < dst_blocks.size(); ++i) {
    if (!CopyBatchNormState(dst_blocks[i].norm1(), src_blocks[i].norm1()) ||
        !CopyBatchNormState(dst_blocks[i].norm2(), src_blocks[i].norm2())) {
      return false;
    }
  }
  return true;
}

void Evaluator::Predict(const Gomoku &game, float *policy, float &value) {
  game.Encode(input_);
  net_.Forward(input_, output_, /*training=*/false);

  value = output_.values_[0];

  // Legal-move masked softmax.
  float max_logit = -1e30f;
  for (int action = 0; action < Gomoku::kActionNum; ++action) {
    if (game.IsLegal(action)) {
      max_logit = std::max(max_logit, output_.policy_logits_[action]);
    }
  }
  float sum = 0.0f;
  for (int action = 0; action < Gomoku::kActionNum; ++action) {
    if (game.IsLegal(action)) {
      const float p = std::exp(output_.policy_logits_[action] - max_logit);
      policy[action] = p;
      sum += p;
    } else {
      policy[action] = 0.0f;
    }
  }
  if (sum > 0.0f) {
    const float inverse = 1.0f / sum;
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      policy[action] *= inverse;
    }
  } else {
    // Defensive fallback: uniform over legal moves.
    int legal = 0;
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      if (game.IsLegal(action)) ++legal;
    }
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      if (game.IsLegal(action)) policy[action] = 1.0f / legal;
    }
  }
}

// --- EvalCache ---

std::string EvalCache::EncodeKey(const Gomoku &game) {
  std::string key(Gomoku::kCellNum + 2, '\0');
  const int player = game.current_player();
  const auto &board = game.board();
  key[0] = static_cast<char>(player);
  // +1 keeps -1 (no last move) distinguishable from action 0.
  key[1] = static_cast<char>(game.last_action() + 1);
  for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
    // 0 empty, 1 own stone, 2 opponent stone (state is player-relative)
    key[cell + 2] = board[cell] == 0
                        ? '\0'
                        : static_cast<char>(board[cell] == player ? 1 : 2);
  }
  return key;
}

bool EvalCache::Lookup(const Gomoku &game, Entry &entry) {
  const std::string key = EncodeKey(game);
  const int shard = static_cast<int>(std::hash<std::string>{}(key) % kShardNum);
  std::lock_guard<std::mutex> lock(mutex_[shard]);
  ++lookups_;
  auto it = shards_[shard].find(key);
  if (it == shards_[shard].end()) {
    return false;
  }
  ++hits_;
  entry = it->second;
  return true;
}

void EvalCache::Store(const Gomoku &game, const float *policy, float value) {
  const std::string key = EncodeKey(game);
  const int shard = static_cast<int>(std::hash<std::string>{}(key) % kShardNum);
  Entry entry;
  std::copy(policy, policy + Gomoku::kActionNum, entry.policy.begin());
  entry.value = value;
  std::lock_guard<std::mutex> lock(mutex_[shard]);
  shards_[shard].emplace(std::move(key), std::move(entry));
}

void EvalCache::Clear() {
  for (int shard = 0; shard < kShardNum; ++shard) {
    std::lock_guard<std::mutex> lock(mutex_[shard]);
    shards_[shard].clear();
  }
  lookups_ = 0;
  hits_ = 0;
}

std::size_t EvalCache::Size() const {
  std::size_t total = 0;
  for (int shard = 0; shard < kShardNum; ++shard) {
    std::lock_guard<std::mutex> lock(mutex_[shard]);
    total += shards_[shard].size();
  }
  return total;
}

} // namespace az
