#include "train/trainer.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace az {

namespace {

// opponent plane has a run of exactly-at-least `len` somewhere
bool OppHasRun(const float *opp_plane, int len) {
  static const int dx[4] = {0, 1, 1, 1};
  static const int dy[4] = {1, 0, 1, -1};
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      for (int d = 0; d < 4; ++d) {
        int k = 0;
        for (int s = 0; s < len; ++s) {
          const int rr = r + s * dx[d], cc = c + s * dy[d];
          if (rr < 0 || rr >= 15 || cc < 0 || cc >= 15) break;
          if (opp_plane[rr * 15 + cc] > 0.5f) ++k; else break;
        }
        if (k >= len) return true;
      }
    }
  }
  return false;
}

} // namespace

void Trainer::RefreshHardSet() {
  hard_indices_.clear();
  const std::size_t n = buffer_.Size();
  hard_indices_.reserve(n / 8);
  for (std::size_t i = 0; i < n; ++i) {
    const Sample &s = buffer_.At(static_cast<int>(i));
    const float *opp = s.planes.data() + 225;
    // "must defend": opponent has a four (near-certain loss if ignored),
    // or an open three in a lost game. Purely data-derived, no play heuristic
    // enters the training signal — this only re-weights SELF-PLAY data.
    if (OppHasRun(opp, 4) || (OppHasRun(opp, 3) && s.value < -0.5f)) {
      hard_indices_.push_back(static_cast<int>(i));
    }
  }
}

bool Trainer::Init(const TrainConfig &config) {
  config_ = config;
  mkdir(config_.run_dir_.c_str(), 0755);
  log_path_ = config_.run_dir_ + "/train.log";

  config_.net_.thread_num_ = 4; // trainer may use the global pool
  if (net_.Init(config_.net_) != deeplearning::PolicyValueResNet::SUCCESS) {
    std::fprintf(stderr, "[trainer] net init failed: %s\n",
                 net_.err_msg().c_str());
    return false;
  }

  deeplearning::PolicyValueLoss::Config loss_config;
  loss_config.policy_size_ = Gomoku::kActionNum;
  loss_config.policy_weight_ = 1.0f;
  loss_config.value_weight_ = config_.value_weight_;
  if (loss_.Init(loss_config) != deeplearning::PolicyValueLoss::SUCCESS) {
    std::fprintf(stderr, "[trainer] loss init failed: %s\n",
                 loss_.err_msg().c_str());
    return false;
  }

  deeplearning::FloatAdamW::Config optimizer_config;
  optimizer_config.weight_decay_ = config_.weight_decay_;
  std::vector<std::size_t> sizes;
  for (const auto &p : net_.TrainableParameters()) {
    sizes.push_back(p.value_->size());
  }
  if (optimizer_.Init(sizes, optimizer_config) !=
      deeplearning::FloatAdamW::SUCCESS) {
    std::fprintf(stderr, "[trainer] optimizer init failed: %s\n",
                 optimizer_.err_msg().c_str());
    return false;
  }

  buffer_.SetCapacity(config_.buffer_capacity_);
  rng_.seed(static_cast<unsigned>(config_.seed_));

  if (config_.resume_) {
    TryResume();
  }
  std::fprintf(stderr, "[trainer] init ok, params=%zu, resume iteration=%d\n",
               net_.parameter_count(), iteration_);
  return true;
}

bool Trainer::TryResume() {
  const std::string net_path = config_.run_dir_ + "/latest.net";
  if (net_.Load(net_path) == deeplearning::PolicyValueResNet::SUCCESS) {
    const std::string opt_path = config_.run_dir_ + "/latest.opt";
    optimizer_.Load(opt_path);
    const std::string state_path = config_.run_dir_ + "/latest.state";
    FILE *in = std::fopen(state_path.c_str(), "rb");
    if (in != nullptr) {
      std::fread(&iteration_, sizeof(iteration_), 1, in);
      std::fread(&global_step_, sizeof(global_step_), 1, in);
      std::fclose(in);
    }
    buffer_.Load(config_.run_dir_ + "/buffer.bin");
    return true;
  }
  return false;
}

void Trainer::Checkpoint(const std::string &tag) {
  net_.Save(config_.run_dir_ + "/" + tag + ".net");
  if (tag == "latest") {
    optimizer_.Save(config_.run_dir_ + "/latest.opt");
    FILE *out = std::fopen((config_.run_dir_ + "/latest.state").c_str(), "wb");
    if (out != nullptr) {
      std::fwrite(&iteration_, sizeof(iteration_), 1, out);
      std::fwrite(&global_step_, sizeof(global_step_), 1, out);
      std::fclose(out);
    }
  }
}

void Trainer::WriteLog(const char *fmt, ...) {
  char line[2048];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(line, sizeof(line), fmt, args);
  va_end(args);
  std::fprintf(stderr, "%s\n", line);
  FILE *out = std::fopen(log_path_.c_str(), "a");
  if (out != nullptr) {
    std::fprintf(out, "%s\n", line);
    std::fclose(out);
  }
}

bool Trainer::TrainStep(int batch_size, float lr, float &policy_loss,
                        float &value_loss) {
  if (buffer_.Size() < static_cast<std::size_t>(batch_size)) {
    return false;
  }
  std::vector<int> indices;
  // 60% of each batch from exponentially recent samples (mean age ~8k):
  // counters stale-data dilution; the rest stays uniform to preserve coverage.
  buffer_.SampleIndicesRecency(batch_size, rng_, 0.6, 8000.0, indices);
  // Hard-negative mining: replace ~hard_fraction_ of the batch with
  // must-defend states (opponent has four, or open three in lost games).
  // All data still comes from self-play; this only re-weights WHICH of our
  // own mistakes get repeated in training.
  if (!hard_indices_.empty()) {
    const int hard_want =
        static_cast<int>(batch_size * hard_fraction_);
    std::uniform_int_distribution<int> pick(
        0, static_cast<int>(hard_indices_.size()) - 1);
    for (int b = 0; b < hard_want; ++b) {
      indices[indices.size() - 1 - b] = hard_indices_[pick(rng_)];
    }
  }

  // Build batch tensors with one random symmetry per sample.
  deeplearning::FloatTensor4D input;
  input.Resize(batch_size, Gomoku::kPlaneNum, Gomoku::kBoardSize,
               Gomoku::kBoardSize, 0.0f);
  std::vector<float> policy_target(batch_size * Gomoku::kActionNum, 0.0f);
  std::vector<std::uint8_t> legal_mask(batch_size * Gomoku::kActionNum, 0);
  std::vector<float> value_target(batch_size, 0.0f);

  std::uniform_int_distribution<int> sym_dist(0, Gomoku::kSymmetryNum - 1);
  std::vector<float> tmp_planes(Gomoku::kPlaneNum * Gomoku::kCellNum);
  std::vector<float> tmp_policy(Gomoku::kActionNum);
  std::vector<float> transformed(Gomoku::kPlaneNum * Gomoku::kCellNum);
  std::vector<float> transformed_policy(Gomoku::kActionNum);

  for (int b = 0; b < batch_size; ++b) {
    const Sample &sample = buffer_.At(indices[b]);
    const int symmetry = sym_dist(rng_);
    Gomoku::TransformPlanes(sample.planes.data(), transformed.data(), symmetry);
    Gomoku::TransformPolicy(sample.policy.data(), transformed_policy.data(),
                            symmetry);
    std::copy(transformed.begin(), transformed.end(),
              input.data() +
                  static_cast<std::size_t>(b) * Gomoku::kPlaneNum *
                      Gomoku::kCellNum);
    std::copy(transformed_policy.begin(), transformed_policy.end(),
              policy_target.data() +
                  static_cast<std::size_t>(b) * Gomoku::kActionNum);
    value_target[b] = sample.value;
    for (int a = 0; a < Gomoku::kActionNum; ++a) {
      // legal iff own/opp planes are both empty at that cell after transform
      const float occupied =
          transformed[a] + transformed[Gomoku::kCellNum + a];
      legal_mask[b * Gomoku::kActionNum + a] = occupied < 0.5f ? 1 : 0;
    }
  }

  deeplearning::PolicyValueResNet::Output output;
  if (net_.Forward(input, output, /*training=*/true) !=
      deeplearning::PolicyValueResNet::SUCCESS) {
    std::fprintf(stderr, "[trainer] forward failed: %s\n",
                 net_.err_msg().c_str());
    return false;
  }

  deeplearning::PolicyValueLoss::Result result;
  if (loss_.ForwardBackward(output.policy_logits_, output.values_, legal_mask,
                            policy_target, value_target, batch_size,
                            result) != deeplearning::PolicyValueLoss::SUCCESS) {
    std::fprintf(stderr, "[trainer] loss failed: %s\n",
                 loss_.err_msg().c_str());
    return false;
  }
  policy_loss = result.policy_loss_;
  value_loss = result.value_loss_;

  deeplearning::FloatTensor4D grad_input; // unused by caller
  if (net_.Backward(result.grad_policy_logits_, result.grad_values_,
                    grad_input) != deeplearning::PolicyValueResNet::SUCCESS) {
    std::fprintf(stderr, "[trainer] backward failed: %s\n",
                 net_.err_msg().c_str());
    return false;
  }

  // FloatAdamW: build parameter view in the SAME order as Init().
  auto params = net_.TrainableParameters();
  std::vector<deeplearning::FloatAdamW::Parameter> adamw_params;
  adamw_params.reserve(params.size());
  for (auto &p : params) {
    adamw_params.push_back({p.value_, p.gradient_, p.apply_weight_decay_});
  }
  if (optimizer_.Step(adamw_params, lr) != deeplearning::FloatAdamW::SUCCESS) {
    std::fprintf(stderr, "[trainer] step failed: %s\n",
                 optimizer_.err_msg().c_str());
    return false;
  }

  net_.ZeroGrad();
  ++global_step_;
  return true;
}

void Trainer::Run() {
  for (; config_.iterations_ < 0 || iteration_ < config_.iterations_;) {
    ++iteration_;
    WriteLog("{\"iter\":%d,\"phase\":\"selfplay_begin\",\"step\":%d}",
             iteration_, global_step_);

    // self-play with latest weights
    RefreshHardSet();
    WriteLog("{\"iter\":%d,\"phase\":\"hard_set\",\"size\":%zu}", iteration_,
             hard_indices_.size());
    config_.selfplay_.hard_seed_indices_ = &hard_indices_;
    SelfPlayStats sp = RunSelfPlay(net_, config_.selfplay_, buffer_);
    const double avg_moves =
        sp.games > 0 ? sp.moves_total / sp.games : 0.0;
    WriteLog("{\"iter\":%d,\"phase\":\"selfplay_done\",\"games\":%d,"
             "\"avg_moves\":%.1f,\"cache_size\":%.0f,"
             "\"cache_hit_rate\":%.3f,\"buffer\":%zu}",
             iteration_, sp.games, avg_moves, sp.eval_cache_size,
             sp.eval_cache_hit_rate, buffer_.Size());

    // training steps
    double policy_loss_sum = 0.0, value_loss_sum = 0.0;
    const int steps = config_.train_steps_;
    int done_steps = 0;
    for (int s = 0; s < steps; ++s) {
      float pl = 0.0f, vl = 0.0f;
      if (!TrainStep(config_.batch_size_, config_.learning_rate_, pl, vl)) {
        break;
      }
      policy_loss_sum += pl;
      value_loss_sum += vl;
      ++done_steps;
    }
    if (done_steps > 0) {
      WriteLog("{\"iter\":%d,\"phase\":\"train_done\",\"steps\":%d,"
               "\"policy_loss\":%.4f,\"value_loss\":%.4f,\"step\":%d}",
               iteration_, done_steps, policy_loss_sum / done_steps,
               value_loss_sum / done_steps, global_step_);
    }

    Checkpoint("latest");
    if (iteration_ == 1 || iteration_ % 5 == 0) {
      Checkpoint("iter" + std::to_string(iteration_));
    }

    // arena gating: challenger (latest) vs best
    if (iteration_ % config_.gate_every_ == 0) {
      deeplearning::PolicyValueResNet best;
      auto arena_config = config_.net_;
      arena_config.thread_num_ = 1;
      best.Init(arena_config);
      const std::string best_path = config_.run_dir_ + "/best.net";
      const bool best_exists =
          best.Load(best_path) == deeplearning::PolicyValueResNet::SUCCESS;

      // quick progress probe vs random baseline
      DuelStats vs_random =
          RunVsRandom(net_, config_.selfplay_.mcts_, 20,
                      config_.selfplay_.worker_num_,
                      config_.selfplay_.max_moves_, config_.seed_ + iteration_);
      WriteLog("{\"iter\":%d,\"phase\":\"eval_random\",\"wins\":%d,"
               "\"losses\":%d,\"draws\":%d}",
               iteration_, vs_random.a_wins, vs_random.b_wins,
               vs_random.draws);

      if (!best_exists) {
        Checkpoint("best"); // first iteration becomes the baseline
        WriteLog("{\"iter\":%d,\"phase\":\"gate\",\"result\":\"init_best\"}",
                 iteration_);
      } else {
        MctsConfig arena_mcts = config_.selfplay_.mcts_;
        arena_mcts.dirichlet_epsilon_ = 0.0f;
        DuelStats duel =
            RunDuel(net_, best, arena_mcts, config_.gate_games_,
                    config_.selfplay_.worker_num_,
                    config_.selfplay_.temperature_move_cutoff_,
                    config_.selfplay_.max_moves_, config_.seed_ + iteration_);
        const int decisive = duel.a_wins + duel.b_wins;
        const double rate = decisive > 0
                                ? static_cast<double>(duel.a_wins) / decisive
                                : 0.5;
        WriteLog("{\"iter\":%d,\"phase\":\"gate\",\"challenger_wins\":%d,"
                 "\"best_wins\":%d,\"draws\":%d,\"rate\":%.3f}",
                 iteration_, duel.a_wins, duel.b_wins, duel.draws, rate);
        if (rate >= config_.gate_threshold_) {
          Checkpoint("best");
          WriteLog("{\"iter\":%d,\"phase\":\"gate\",\"result\":"
                   "\"promoted\",\"rate\":%.3f}",
                   iteration_, rate);
        }
      }
    }

    if (iteration_ % config_.save_buffer_every_ == 0) {
      buffer_.Save(config_.run_dir_ + "/buffer.bin");
    }
  }
}

} // namespace az
