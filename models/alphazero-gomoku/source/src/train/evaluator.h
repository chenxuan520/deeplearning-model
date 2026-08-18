#pragma once

#include "cnn/policy_value_resnet.h"
#include "game/gomoku.h"

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace az {

// Interface MCTS uses for position evaluation. The neural-net Evaluator and
// test doubles both implement this.
class INetEvaluator {
public:
  virtual ~INetEvaluator() = default;
  // policy output: probabilities over all kActionNum actions (0 on illegal
  // moves, sums to 1 over legal ones). value: expected game outcome in
  // [-1, 1] from the perspective of the player to move.
  virtual void Predict(const Gomoku &game, float *policy, float &value) = 0;
};

// Copies all trainable parameters AND batch-norm running statistics (same
// order, identical configs required). Returns false on any shape mismatch.
bool AssignWeights(deeplearning::PolicyValueResNet &dst,
                   deeplearning::PolicyValueResNet &src);

// Neural-net evaluator. NOT thread-safe: give every worker its own copy.
class Evaluator : public INetEvaluator {
public:
  bool Init(const deeplearning::PolicyValueResNet::Config &config) {
    return net_.Init(config) == deeplearning::PolicyValueResNet::SUCCESS;
  }
  void Predict(const Gomoku &game, float *policy, float &value) override;

  deeplearning::PolicyValueResNet &net() { return net_; }

private:
  deeplearning::PolicyValueResNet net_;
  deeplearning::FloatTensor4D input_;
  deeplearning::PolicyValueResNet::Output output_;
};

// Thread-safe evaluation cache: exact board-position key (this player's /
// opponent's stone layout as seen by the player to move) -> net output.
// Valid only while net weights are fixed; Clear() after every weight update.
// Pure memory-for-CPU: repeated positions across concurrent games skip the
// expensive forward pass.
class EvalCache {
public:
  struct Entry {
    std::array<float, Gomoku::kActionNum> policy;
    float value;
  };

  bool Lookup(const Gomoku &game, Entry &entry);
  void Store(const Gomoku &game, const float *policy, float value);
  void Clear();
  std::size_t Size() const;

  // stats for logs
  std::size_t Lookups() const { return lookups_; }
  std::size_t Hits() const { return hits_; }

private:
  static constexpr int kShardNum = 64;
  static std::string EncodeKey(const Gomoku &game);

  mutable std::mutex mutex_[kShardNum];
  std::unordered_map<std::string, Entry> shards_[kShardNum];
  std::size_t lookups_ = 0;
  std::size_t hits_ = 0;
};

// Decorator adding cache behavior on top of another evaluator.
class CachedEvaluator : public INetEvaluator {
public:
  CachedEvaluator(INetEvaluator *inner, EvalCache *cache)
      : inner_(inner), cache_(cache) {}
  void Predict(const Gomoku &game, float *policy, float &value) override {
    if (cache_ != nullptr) {
      EvalCache::Entry entry;
      if (cache_->Lookup(game, entry)) {
        std::copy(entry.policy.begin(), entry.policy.end(), policy);
        value = entry.value;
        return;
      }
    }
    inner_->Predict(game, policy, value);
    if (cache_ != nullptr) {
      cache_->Store(game, policy, value);
    }
  }

private:
  INetEvaluator *inner_;
  EvalCache *cache_;
};

} // namespace az
