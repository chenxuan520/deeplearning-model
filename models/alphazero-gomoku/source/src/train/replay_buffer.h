#pragma once

#include "game/gomoku.h"

#include <array>
#include <cstddef>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace az {

// One training record: encoded planes (current-player perspective), the MCTS
// visit distribution over all actions, and the final game outcome z from the
// perspective of the player to move at this state.
struct Sample {
  std::array<float, Gomoku::kPlaneNum * Gomoku::kCellNum> planes;
  std::array<float, Gomoku::kActionNum> policy;
  float value = 0.0f;
};

// Fixed-capacity ring buffer. Push is thread-safe (self-play workers share
// one buffer); sampling happens on the trainer thread only.
class ReplayBuffer {
public:
  explicit ReplayBuffer(std::size_t capacity = 200000) : capacity_(capacity) {
    data_.resize(capacity);
  }

  // Re-sizes capacity; only valid while the buffer is empty (before any Push
  // or Load). Exists because the embedded mutex makes ReplayBuffer
  // non-assignable.
  void SetCapacity(std::size_t capacity) {
    capacity_ = capacity;
    data_.clear();
    data_.resize(capacity);
    write_pos_ = 0;
    size_ = 0;
  }

  void Push(const Sample &sample);
  // Fills `indices` with uniform-random sample indices (with replacement).
  void SampleIndices(int count, std::mt19937 &rng,
                     std::vector<int> &indices) const;
  // Recency-weighted sampling: with probability p_recent, draws an
  // exponentially-decaying index biased to recent samples (mean age =
  // 1/lambda). Otherwise uniform over the whole buffer. Prevents legacy
  // low-quality-era samples from diluting fresh signals.
  void SampleIndicesRecency(int count, std::mt19937 &rng,
                            double p_recent, double mean_age,
                            std::vector<int> &indices) const;
  const Sample &At(int index) const { return data_[index]; }

  std::size_t Size() const;
  std::size_t capacity() const { return capacity_; }

  bool Save(const std::string &file) const;
  bool Load(const std::string &file);

private:
  std::size_t capacity_;
  std::vector<Sample> data_;
  std::size_t write_pos_ = 0;
  std::size_t size_ = 0;
  mutable std::mutex mutex_;
};

} // namespace az
