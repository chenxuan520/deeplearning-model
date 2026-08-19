#include "train/trainer.h"

#include <cstdarg>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace az {

namespace {

bool SyncFile(const std::string &path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) return false;
  const bool ok = ::fsync(descriptor) == 0;
  ::close(descriptor);
  return ok;
}

bool SyncParentDirectory(const std::string &path) {
  const std::size_t slash = path.find_last_of('/');
  const std::string directory =
      slash == std::string::npos ? "." : path.substr(0, slash);
  const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0) return false;
  const bool ok = ::fsync(descriptor) == 0;
  ::close(descriptor);
  return ok;
}

bool AtomicReplace(const std::string &temporary, const std::string &target) {
  if (!SyncFile(temporary) ||
      std::rename(temporary.c_str(), target.c_str()) != 0) {
    std::remove(temporary.c_str());
    return false;
  }
  // Once rename succeeds the target is visible. If the directory cannot be
  // synced, fail-stop before deleting the previous generation: after reboot
  // either the old or new pointer is valid and both bundles still exist.
  if (!SyncParentDirectory(target)) std::abort();
  return true;
}

bool ReadPlateauMarker(const std::string &path, std::string &request_id,
                       int &iteration) {
  char id[128] = {};
  FILE *in = std::fopen(path.c_str(), "r");
  if (in == nullptr) return false;
  const bool ok = std::fscanf(in, "%127s %d", id, &iteration) == 2;
  std::fclose(in);
  if (ok) request_id = id;
  return ok;
}

bool ReadGenerationPointer(const std::string &run_dir, int &generation,
                           int &best_generation, int &buffer_generation) {
  std::ifstream input(run_dir + "/latest.current");
  if (!input.good()) return false;
  std::string content((std::istreambuf_iterator<char>(input)),
                      std::istreambuf_iterator<char>());
  std::istringstream parser(content);
  std::vector<std::string> tokens;
  for (std::string token; parser >> token;) tokens.push_back(token);
  if (tokens.size() != 3) return false;
  auto parse_int = [](const std::string &token, int &value) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(token.c_str(), &end, 10);
    if (errno != 0 || end == token.c_str() || *end != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX)
      return false;
    value = static_cast<int>(parsed);
    return true;
  };
  if (!parse_int(tokens[0], generation) || generation < 0) return false;
  if (!parse_int(tokens[1], best_generation) || best_generation < -1)
    return false;
  if (!parse_int(tokens[2], buffer_generation) || buffer_generation < -1)
    return false;
  return true;
}

bool HasVersionedCheckpointFiles(const std::string &run_dir) {
  DIR *directory = ::opendir(run_dir.c_str());
  if (directory == nullptr) return false;
  bool found = false;
  while (const dirent *entry = ::readdir(directory)) {
    if (std::strncmp(entry->d_name, "checkpoint.latest.", 18) == 0) {
      found = true;
      break;
    }
  }
  ::closedir(directory);
  return found;
}

std::string GenerationPrefix(const std::string &run_dir, int generation) {
  return run_dir + "/checkpoint.latest." + std::to_string(generation);
}

std::string BestGenerationPath(const std::string &run_dir, int generation) {
  return run_dir + "/checkpoint.best." + std::to_string(generation) + ".net";
}

std::string BufferGenerationPath(const std::string &run_dir, int generation) {
  return run_dir + "/checkpoint.buffer." + std::to_string(generation) + ".bin";
}

void PublishCompatibilityAlias(const std::string &source,
                               const std::string &target) {
  const std::string temporary = target + ".link.tmp";
  std::remove(temporary.c_str());
  if (::link(source.c_str(), temporary.c_str()) == 0)
    AtomicReplace(temporary, target);
}

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
    if (!TryResume()) {
      struct stat checkpoint_stat {};
      const bool checkpoint_exists =
          ::stat((config_.run_dir_ + "/latest.current").c_str(),
                 &checkpoint_stat) == 0 ||
          ::stat((config_.run_dir_ + "/latest.net").c_str(),
                 &checkpoint_stat) == 0 ||
          ::stat((config_.run_dir_ + "/latest.versioned").c_str(),
                 &checkpoint_stat) == 0 ||
          HasVersionedCheckpointFiles(config_.run_dir_);
      if (checkpoint_exists) {
        std::fprintf(stderr, "[trainer] existing checkpoint is incomplete\n");
        return false;
      }
    }
  }
  std::fprintf(stderr, "[trainer] init ok, params=%zu, resume iteration=%d\n",
               net_.parameter_count(), iteration_);
  return true;
}

bool Trainer::TryResume() {
  int generation = -1;
  int best_generation = -1;
  int buffer_generation = -1;
  struct stat pointer_stat {};
  const bool pointer_exists =
      ::stat((config_.run_dir_ + "/latest.current").c_str(),
             &pointer_stat) == 0;
  struct stat versioned_stat {};
  const bool versioned_marker_exists =
      ::stat((config_.run_dir_ + "/latest.versioned").c_str(),
             &versioned_stat) == 0;
  if (!pointer_exists && versioned_marker_exists) return false;
  if (pointer_exists &&
      !ReadGenerationPointer(config_.run_dir_, generation, best_generation,
                             buffer_generation))
    return false;
  const bool versioned = pointer_exists;
  const std::string prefix = versioned
      ? GenerationPrefix(config_.run_dir_, generation)
      : config_.run_dir_ + "/latest";
  const std::string net_path = prefix + ".net";
  if (net_.Load(net_path) == deeplearning::PolicyValueResNet::SUCCESS) {
    const std::string opt_path = prefix + ".opt";
    if (optimizer_.Load(opt_path) != deeplearning::FloatAdamW::SUCCESS)
      return false;
    const std::string state_path = prefix + ".state";
    FILE *in = std::fopen(state_path.c_str(), "rb");
    const bool state_ok = in != nullptr &&
        std::fread(&iteration_, sizeof(iteration_), 1, in) == 1 &&
        std::fread(&global_step_, sizeof(global_step_), 1, in) == 1;
    if (in != nullptr) std::fclose(in);
    if (!state_ok || (versioned && iteration_ != generation)) return false;
    current_best_generation_ = best_generation;
    if (!versioned && best_generation < 0) {
      const std::string legacy_best = config_.run_dir_ + "/best.net";
      struct stat legacy_best_stat {};
      if (::stat(legacy_best.c_str(), &legacy_best_stat) == 0) {
        const std::string migrated_best =
            BestGenerationPath(config_.run_dir_, 0);
        PublishCompatibilityAlias(legacy_best, migrated_best);
        struct stat migrated_stat {};
        if (::stat(migrated_best.c_str(), &migrated_stat) != 0)
          return false;
        current_best_generation_ = 0;
      }
    }
    if (best_generation >= 0) {
      deeplearning::PolicyValueResNet best;
      auto best_config = config_.net_;
      best_config.thread_num_ = 1;
      if (best.Init(best_config) != deeplearning::PolicyValueResNet::SUCCESS ||
          best.Load(BestGenerationPath(config_.run_dir_, best_generation)) !=
              deeplearning::PolicyValueResNet::SUCCESS)
        return false;
    }
    current_buffer_generation_ = buffer_generation;
    if (versioned) {
      if (buffer_generation >= 0) {
        const std::string buffer_path =
            BufferGenerationPath(config_.run_dir_, buffer_generation);
        struct stat buffer_stat {};
        if (::stat(buffer_path.c_str(), &buffer_stat) != 0 ||
            !buffer_.Load(buffer_path))
          return false;
      }
    } else {
      const std::string buffer_path = config_.run_dir_ + "/buffer.bin";
      struct stat buffer_stat {};
      if (::stat(buffer_path.c_str(), &buffer_stat) == 0 &&
          !buffer_.Load(buffer_path))
        return false;
    }
    return true;
  }
  return false;
}

bool Trainer::Checkpoint(const std::string &tag) {
  const bool latest = tag == "latest";
  int old_generation = -1;
  int ignored_best_generation = -1;
  int ignored_buffer_generation = -1;
  if (latest) {
    DiscardPendingLatest();
    ReadGenerationPointer(config_.run_dir_, old_generation,
                          ignored_best_generation,
                          ignored_buffer_generation);
  }
  const std::string prefix = latest
      ? GenerationPrefix(config_.run_dir_, iteration_)
      : config_.run_dir_ + "/" + tag;
  const std::string net_path = prefix + ".net";
  const std::string net_tmp = net_path + ".tmp";
  if (net_.Save(net_tmp) != deeplearning::PolicyValueResNet::SUCCESS)
    return false;
  if (!latest) return AtomicReplace(net_tmp, net_path);

  const std::string opt_path = prefix + ".opt";
  const std::string opt_tmp = opt_path + ".tmp";
  if (optimizer_.Save(opt_tmp) != deeplearning::FloatAdamW::SUCCESS) {
    std::remove(net_tmp.c_str());
    return false;
  }
  const std::string state_path = prefix + ".state";
  const std::string state_tmp = state_path + ".tmp";
  FILE *out = std::fopen(state_tmp.c_str(), "wb");
  const bool state_ok =
      out != nullptr &&
      std::fwrite(&iteration_, sizeof(iteration_), 1, out) == 1 &&
      std::fwrite(&global_step_, sizeof(global_step_), 1, out) == 1 &&
      std::fflush(out) == 0;
  if (out != nullptr) std::fclose(out);
  if (!state_ok) {
    std::remove(net_tmp.c_str());
    std::remove(opt_tmp.c_str());
    std::remove(state_tmp.c_str());
    return false;
  }
  // Publish all immutable generation files first. None becomes authoritative
  // until the single pointer below is atomically replaced.
  const bool net_published = AtomicReplace(net_tmp, net_path);
  const bool opt_published =
      net_published && AtomicReplace(opt_tmp, opt_path);
  const bool state_published =
      opt_published && AtomicReplace(state_tmp, state_path);
  if (!state_published) {
    std::remove(net_tmp.c_str());
    std::remove(opt_tmp.c_str());
    std::remove(state_tmp.c_str());
    return false;
  }

  pending_latest_generation_ = iteration_;
  previous_latest_generation_ = old_generation;
  return true;
}

bool Trainer::PrepareBestCheckpoint() {
  DiscardPendingBest();
  const std::string path = BestGenerationPath(config_.run_dir_, iteration_);
  const std::string temporary = path + ".tmp";
  if (net_.Save(temporary) != deeplearning::PolicyValueResNet::SUCCESS ||
      !AtomicReplace(temporary, path))
    return false;
  pending_best_generation_ = iteration_;
  return true;
}

bool Trainer::PublishLatestCheckpoint() {
  if (pending_latest_generation_ < 0) return false;
  const int generation = pending_latest_generation_;
  const int old_generation = previous_latest_generation_;
  const int old_best_generation = current_best_generation_;
  const int old_buffer_generation = current_buffer_generation_;
  const int committed_best_generation = pending_best_generation_ >= 0
      ? pending_best_generation_ : current_best_generation_;
  const int committed_buffer_generation = pending_buffer_generation_ >= 0
      ? pending_buffer_generation_ : current_buffer_generation_;
  const std::string prefix = GenerationPrefix(config_.run_dir_, generation);
  const std::string net_path = prefix + ".net";
  const std::string opt_path = prefix + ".opt";
  const std::string state_path = prefix + ".state";
  const std::string pointer = config_.run_dir_ + "/latest.current";
  const std::string pointer_tmp = pointer + ".tmp";
  FILE *pointer_out = std::fopen(pointer_tmp.c_str(), "w");
  const bool pointer_ok = pointer_out != nullptr &&
      std::fprintf(pointer_out, "%d %d %d\n", generation,
                   committed_best_generation,
                   committed_buffer_generation) > 0 &&
      std::fflush(pointer_out) == 0;
  if (pointer_out != nullptr) std::fclose(pointer_out);
  if (!pointer_ok || !AtomicReplace(pointer_tmp, pointer)) {
    DiscardPendingLatest();
    return false;
  }
  const std::string versioned_marker =
      config_.run_dir_ + "/latest.versioned";
  struct stat marker_stat {};
  if (::stat(versioned_marker.c_str(), &marker_stat) != 0) {
    const std::string marker_tmp = versioned_marker + ".tmp";
    FILE *marker = std::fopen(marker_tmp.c_str(), "w");
    const bool marker_ok = marker != nullptr &&
        std::fputs("1\n", marker) >= 0 && std::fflush(marker) == 0;
    if (marker != nullptr) std::fclose(marker);
    if (!marker_ok || !AtomicReplace(marker_tmp, versioned_marker))
      std::abort();
  }

  // Keep the historical paths for inference scripts and humans. Resume and
  // plateau logic use latest.current, so these aliases are not transactional
  // state.
  PublishCompatibilityAlias(net_path, config_.run_dir_ + "/latest.net");
  PublishCompatibilityAlias(opt_path, config_.run_dir_ + "/latest.opt");
  PublishCompatibilityAlias(state_path, config_.run_dir_ + "/latest.state");
  if (committed_best_generation >= 0) {
    PublishCompatibilityAlias(
        BestGenerationPath(config_.run_dir_, committed_best_generation),
        config_.run_dir_ + "/best.net");
  }
  if (committed_buffer_generation >= 0) {
    PublishCompatibilityAlias(
        BufferGenerationPath(config_.run_dir_, committed_buffer_generation),
        config_.run_dir_ + "/buffer.bin");
  }
  if (old_generation >= 0 && old_generation != iteration_) {
    const std::string old = GenerationPrefix(config_.run_dir_, old_generation);
    std::remove((old + ".net").c_str());
    std::remove((old + ".opt").c_str());
    std::remove((old + ".state").c_str());
    SyncParentDirectory(pointer);
  }
  if (old_best_generation >= 0 &&
      old_best_generation != committed_best_generation)
    std::remove(BestGenerationPath(config_.run_dir_, old_best_generation).c_str());
  current_best_generation_ = committed_best_generation;
  if (old_buffer_generation >= 0 &&
      old_buffer_generation != committed_buffer_generation)
    std::remove(BufferGenerationPath(config_.run_dir_,
                                     old_buffer_generation).c_str());
  current_buffer_generation_ = committed_buffer_generation;
  pending_latest_generation_ = -1;
  previous_latest_generation_ = -1;
  pending_best_generation_ = -1;
  pending_buffer_generation_ = -1;
  return true;
}

void Trainer::DiscardPendingBest() {
  if (pending_best_generation_ < 0) return;
  std::remove(BestGenerationPath(config_.run_dir_,
                                 pending_best_generation_).c_str());
  std::remove((BestGenerationPath(config_.run_dir_,
                                  pending_best_generation_) + ".tmp").c_str());
  pending_best_generation_ = -1;
}

void Trainer::DiscardPendingLatest() {
  if (pending_latest_generation_ < 0) return;
  const std::string prefix =
      GenerationPrefix(config_.run_dir_, pending_latest_generation_);
  std::remove((prefix + ".net").c_str());
  std::remove((prefix + ".opt").c_str());
  std::remove((prefix + ".state").c_str());
  std::remove((prefix + ".net.tmp").c_str());
  std::remove((prefix + ".opt.tmp").c_str());
  std::remove((prefix + ".state.tmp").c_str());
  pending_latest_generation_ = -1;
  previous_latest_generation_ = -1;
}

bool Trainer::PrepareReplayBuffer() {
  DiscardPendingReplayBuffer();
  const std::string path = BufferGenerationPath(config_.run_dir_, iteration_);
  const std::string temporary = path + ".tmp";
  if (!buffer_.Save(temporary) || !AtomicReplace(temporary, path)) return false;
  pending_buffer_generation_ = iteration_;
  return true;
}

void Trainer::DiscardPendingReplayBuffer() {
  if (pending_buffer_generation_ < 0) return;
  const std::string path =
      BufferGenerationPath(config_.run_dir_, pending_buffer_generation_);
  std::remove(path.c_str());
  std::remove((path + ".tmp").c_str());
  pending_buffer_generation_ = -1;
}

bool Trainer::WaitForPlateauDecision(bool replay_already_saved) {
  const std::string request = config_.run_dir_ + "/PLATEAU_CHECK_REQUEST";
  std::string request_id;
  int trigger_iteration = -1;
  if (!ReadPlateauMarker(request, request_id, trigger_iteration) ||
      trigger_iteration > iteration_)
    return false;
  if (!replay_already_saved) {
    WriteLog("{\"iter\":%d,\"phase\":\"plateau_pause_deferred\","
             "\"reason\":\"buffer_deferred\"}", iteration_);
    // The request arrived after this iteration's replay-save decision. Keep
    // it: the next iteration will include a replay generation in the atomic
    // pointer and pause safely.
    return false;
  }

  const std::string paused = config_.run_dir_ + "/PLATEAU_PAUSED";
  const std::string paused_tmp = paused + ".tmp";
  FILE *out = std::fopen(paused_tmp.c_str(), "w");
  const bool pause_written = out != nullptr &&
      std::fprintf(out, "%s %d\n", request_id.c_str(), iteration_) > 0 &&
      std::fflush(out) == 0;
  if (out != nullptr) std::fclose(out);
  if (!pause_written || !AtomicReplace(paused_tmp, paused)) {
    WriteLog("{\"iter\":%d,\"phase\":\"plateau_pause_failed\","
             "\"reason\":\"pause_marker\"}", iteration_);
    std::remove(request.c_str());
    return false;
  }
  WriteLog("{\"iter\":%d,\"phase\":\"plateau_paused\",\"step\":%d}",
           iteration_, global_step_);

  const std::string reached = config_.run_dir_ + "/PLATEAU_REACHED";
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::hours(48);
  while (std::chrono::steady_clock::now() < deadline) {
    std::string reached_id;
    int reached_iteration = -1;
    if (ReadPlateauMarker(reached, reached_id, reached_iteration) &&
        reached_id == request_id && reached_iteration == iteration_) {
      WriteLog("{\"iter\":%d,\"phase\":\"plateau_stop_ack\","
               "\"step\":%d}", iteration_, global_step_);
      std::remove(paused.c_str());
      std::remove(request.c_str());
      return true;
    }
    std::string current_request_id;
    int current_trigger = -1;
    if (!ReadPlateauMarker(request, current_request_id, current_trigger) ||
        current_request_id != request_id) {
      WriteLog("{\"iter\":%d,\"phase\":\"plateau_resume\",\"step\":%d}",
               iteration_, global_step_);
      std::remove(paused.c_str());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }
  WriteLog("{\"iter\":%d,\"phase\":\"plateau_resume\","
           "\"reason\":\"monitor_timeout\"}", iteration_);
  std::string current_request_id;
  int current_trigger = -1;
  if (ReadPlateauMarker(request, current_request_id, current_trigger) &&
      current_request_id == request_id)
    std::remove(request.c_str());
  std::remove(paused.c_str());
  return false;
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

    const bool latest_prepared = Checkpoint("latest");
    if (iteration_ == 1 || iteration_ % 5 == 0) {
      if (!Checkpoint("iter" + std::to_string(iteration_))) {
        WriteLog("{\"iter\":%d,\"phase\":\"checkpoint_failed\","
                 "\"tag\":\"iter\"}", iteration_);
      }
    }

    bool gate_saved = true;
    bool gate_init_pending = false;
    bool gate_promotion_pending = false;
    double gate_promotion_rate = 0.0;
    // arena gating: challenger (latest) vs best
    if (iteration_ % config_.gate_every_ == 0) {
      deeplearning::PolicyValueResNet best;
      auto arena_config = config_.net_;
      arena_config.thread_num_ = 1;
      best.Init(arena_config);
      const std::string best_path = current_best_generation_ >= 0
          ? BestGenerationPath(config_.run_dir_, current_best_generation_)
          : config_.run_dir_ + "/best.net";
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
        if (PrepareBestCheckpoint()) { // first iteration becomes baseline
          gate_init_pending = true;
        } else {
          gate_saved = false;
          WriteLog("{\"iter\":%d,\"phase\":\"checkpoint_failed\","
                   "\"tag\":\"best\"}", iteration_);
        }
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
          if (PrepareBestCheckpoint()) {
            gate_promotion_pending = true;
            gate_promotion_rate = rate;
          } else {
            gate_saved = false;
            WriteLog("{\"iter\":%d,\"phase\":\"checkpoint_failed\","
                     "\"tag\":\"best\"}", iteration_);
          }
        }
      }
    }

    std::string pause_request_id;
    int pause_trigger = -1;
    const bool plateau_requested = ReadPlateauMarker(
        config_.run_dir_ + "/PLATEAU_CHECK_REQUEST",
        pause_request_id, pause_trigger) && pause_trigger <= iteration_;
    bool replay_saved = false;
    bool replay_durable = true;
    if (iteration_ % config_.save_buffer_every_ == 0 || plateau_requested ||
        current_buffer_generation_ < 0) {
      replay_saved = PrepareReplayBuffer();
      replay_durable = replay_saved;
      if (!replay_saved) {
        WriteLog("{\"iter\":%d,\"phase\":\"checkpoint_failed\","
                 "\"tag\":\"buffer\"}", iteration_);
      }
    }
    // This is the durable iteration boundary: latest checkpoint is complete,
    // any scheduled gate has finished, and the replay buffer was saved when
    // required. External monitors may only judge/freeze these iterations.
    const bool latest_published =
        latest_prepared && gate_saved && replay_durable &&
        PublishLatestCheckpoint();
    if (latest_published) {
      if (gate_init_pending) {
        WriteLog("{\"iter\":%d,\"phase\":\"gate\","
                 "\"result\":\"init_best\"}", iteration_);
      }
      if (gate_promotion_pending) {
        WriteLog("{\"iter\":%d,\"phase\":\"gate\",\"result\":"
                 "\"promoted\",\"rate\":%.3f}",
                 iteration_, gate_promotion_rate);
      }
      WriteLog("{\"iter\":%d,\"phase\":\"iteration_complete\","
               "\"step\":%d}", iteration_, global_step_);
      if (WaitForPlateauDecision(replay_saved)) return;
    } else {
      DiscardPendingLatest();
      DiscardPendingBest();
      DiscardPendingReplayBuffer();
      WriteLog("{\"iter\":%d,\"phase\":\"checkpoint_failed\","
               "\"tag\":\"latest\"}", iteration_);
    }
  }
}

} // namespace az
