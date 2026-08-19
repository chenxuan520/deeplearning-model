#pragma once

#include "loss/policy_value_loss.h"
#include "optimizer/float_adamw.h"
#include "train/replay_buffer.h"
#include "train/self_play.h"

#include <random>
#include <string>

namespace az {

struct TrainConfig {
  deeplearning::PolicyValueResNet::Config net_;
  SelfPlayConfig selfplay_;

  std::string run_dir_ = "runtime";
  int iterations_ = -1;              // -1 = run forever
  int train_steps_ = 150;            // optimizer steps per iteration
  int batch_size_ = 256;
  float learning_rate_ = 1e-3f;
  float weight_decay_ = 1e-4f;
  float value_weight_ = 1.0f;   // value head loss weight (policy fixed at 1)
  std::size_t buffer_capacity_ = 200000;

  int gate_every_ = 5;               // every N iterations
  int gate_games_ = 20;
  float gate_threshold_ = 0.55f;     // challenger win rate to replace best

  int save_buffer_every_ = 10;
  int seed_ = 42;
  bool resume_ = true;               // resume from runtime/latest.net if found
};

// AlphaZero training loop: self-play -> train steps -> arena gating.
class Trainer {
public:
  bool Init(const TrainConfig &config);
  void Run();

private:
  // One training step over a random minibatch with symmetry augmentation.
  // Returns (policy+value) loss sum for logging.
  bool TrainStep(int batch_size, float learning_rate, float &policy_loss,
                 float &value_loss);
  void RefreshHardSet(); // re-scan buffer once per iteration (hard negatives)
  bool Checkpoint(const std::string &tag);
  bool PrepareBestCheckpoint();
  bool PublishLatestCheckpoint();
  void DiscardPendingLatest();
  void DiscardPendingBest();
  bool PrepareReplayBuffer();
  void DiscardPendingReplayBuffer();
  // Called only at a durable iteration boundary. Returns true when a verified
  // plateau result requests a clean trainer exit.
  bool WaitForPlateauDecision(bool replay_already_saved);
  bool TryResume();
  void WriteLog(const char *fmt, ...);

  TrainConfig config_;
  deeplearning::PolicyValueResNet net_;
  deeplearning::PolicyValueLoss loss_;
  deeplearning::FloatAdamW optimizer_;
  ReplayBuffer buffer_;
  std::mt19937 rng_;
  int iteration_ = 0;
  int global_step_ = 0;
  std::string log_path_;
  std::vector<int> hard_indices_; // buffer indices of "must defend" states
  float hard_fraction_ = 0.3f;    // of each batch
  int pending_latest_generation_ = -1;
  int previous_latest_generation_ = -1;
  int current_best_generation_ = -1;
  int pending_best_generation_ = -1;
  int current_buffer_generation_ = -1;
  int pending_buffer_generation_ = -1;
};

} // namespace az
