#include "train/self_play.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

namespace az {

namespace {

// Reconstructs a game state from a buffer sample's planes (mover-relative:
// own = mover stones, side plane = 1 if mover is black). Returns false on
// inconsistency.
bool GameFromSample(const Sample &sample, Gomoku &game) {
  game.Reset();
  int mover_stones = 0, other_stones = 0;
  int last_action = -1;
  const int side_is_black = sample.planes[3 * 225] > 0.5f ? 1 : 0;
  const int8_t mover = side_is_black ? Gomoku::kBlack : Gomoku::kWhite;
  for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
    if (sample.planes[cell] > 0.5f) {
      game.MutableBoard()[cell] = static_cast<int8_t>(mover);
      ++mover_stones;
    } else if (sample.planes[Gomoku::kCellNum + cell] > 0.5f) {
      game.MutableBoard()[cell] = static_cast<int8_t>(-mover);
      ++other_stones;
    }
    if (sample.planes[2 * Gomoku::kCellNum + cell] > 0.5f) {
      if (last_action >= 0) return false;
      last_action = cell;
    }
  }
  // In a legal position, the mover has same or one fewer stone than opponent
  // when side==black... simplest sanity: total stones even if black to move.
  const int total = mover_stones + other_stones;
  if (side_is_black && total % 2 != 0) return false;
  if (!side_is_black && total % 2 != 1) return false;
  game.SetState(mover, total, /*result=*/0, last_action);
  return true;
}

// Plays a single self-play game; appends samples (with final z values) to the
// buffer. Returns the move count.
int PlaySelfPlayGame(Evaluator &evaluator, EvalCache *cache,
                     const SelfPlayConfig &config, ReplayBuffer &buffer,
                     const ReplayBuffer *seed_source, std::mt19937 &rng) {
  Mcts mcts;
  CachedEvaluator cached(&evaluator, cache);
  INetEvaluator *used = cache != nullptr
                            ? static_cast<INetEvaluator *>(&cached)
                            : static_cast<INetEvaluator *>(&evaluator);

  Gomoku game;
  // Curriculum seeding: optionally start from a hard defensive position.
  if (config.hard_seed_indices_ != nullptr && seed_source != nullptr &&
      !config.hard_seed_indices_->empty()) {
    std::uniform_real_distribution<float> roll(0.0f, 1.0f);
    if (roll(rng) < config.seed_from_hard_prob_) {
      std::uniform_int_distribution<int> pick(
          0, static_cast<int>(config.hard_seed_indices_->size()) - 1);
      const int idx = (*config.hard_seed_indices_)[pick(rng)];
      if (idx >= 0 &&
          static_cast<std::size_t>(idx) < seed_source->Size()) {
        Gomoku seeded;
        if (GameFromSample(seed_source->At(idx), seeded)) {
          game = seeded;
        }
      }
    }
  }
  std::vector<Sample> history;
  std::vector<int> history_player;
  std::vector<int> visit_action, visit_count;
  Sample sample;

  while (!game.IsTerminal() && game.move_count() < config.max_moves_) {
    mcts.Search(game, config.mcts_, *used, rng, visit_action, visit_count);

    Mcts::VisitDistribution(visit_action, visit_count, sample.policy.data());
    game.EncodeInto(sample.planes.data());
    history.push_back(sample);
    history_player.push_back(game.current_player());

    // pick the action: sample during the opening, argmax later
    int action = -1;
    if (game.move_count() < config.temperature_move_cutoff_) {
      std::discrete_distribution<int> dist(sample.policy.begin(),
                                           sample.policy.end());
      action = dist(rng);
    } else {
      action = static_cast<int>(
          std::max_element(sample.policy.begin(), sample.policy.end()) -
          sample.policy.begin());
    }
    // dirichlet noise can spread counts onto anything legal; guard anyway
    if (action < 0 || !game.IsLegal(action)) {
      std::vector<int> legal;
      for (int a = 0; a < Gomoku::kActionNum; ++a)
        if (game.IsLegal(a)) legal.push_back(a);
      std::uniform_int_distribution<int> dist2(0,
                                               (int)legal.size() - 1);
      action = legal[dist2(rng)];
    }
    game.Apply(action);
  }

  const int result = game.IsTerminal() ? game.Result() : 2; // cap -> draw
  const int winner = (result == 2) ? 0 : result;
  for (std::size_t i = 0; i < history.size(); ++i) {
    history[i].value =
        winner == 0 ? 0.0f : (history_player[i] == winner ? 1.0f : -1.0f);
    buffer.Push(history[i]);
  }
  return game.move_count();
}

} // namespace

SelfPlayStats RunSelfPlay(deeplearning::PolicyValueResNet &master,
                          const SelfPlayConfig &config,
                          ReplayBuffer &buffer) {
  EvalCache cache;
  EvalCache *cache_ptr = config.use_cache_ ? &cache : nullptr;

  std::atomic<int> next_game{0};
  std::atomic<int> finished{0};
  SelfPlayStats stats;
  std::mutex stats_mutex;

  auto worker = [&](int worker_index) {
    Evaluator evaluator;
    deeplearning::PolicyValueResNet::Config net_config = master.config();
    net_config.thread_num_ = 1; // workers never share the global pool
    if (!evaluator.Init(net_config)) {
      std::fprintf(stderr, "[selfplay] worker %d net init failed\n",
                   worker_index);
      return;
    }
    if (!AssignWeights(evaluator.net(), master)) {
      std::fprintf(stderr, "[selfplay] worker %d weight copy failed\n",
                   worker_index);
      return;
    }
    std::mt19937 rng(
        static_cast<unsigned>(config.seed_ * 1000003u + worker_index));
    while (true) {
      const int game_index = next_game.fetch_add(1);
      if (game_index >= config.game_num_) {
        break;
      }
      const int moves =
          PlaySelfPlayGame(evaluator, cache_ptr, config, buffer, &buffer,
                           rng);
      std::lock_guard<std::mutex> lock(stats_mutex);
      ++stats.games;
      stats.moves_total += moves;
      const int done = finished.fetch_add(1) + 1;
      if (done % 5 == 0 || done == config.game_num_) {
        std::fprintf(stderr, "\r[selfplay] games %d/%d", done,
                     config.game_num_);
        std::fflush(stderr);
      }
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < config.worker_num_; ++i) {
    threads.emplace_back(worker, i);
  }
  for (auto &t : threads) t.join();
  std::fprintf(stderr, "\n");

  stats.samples = stats.games > 0 ? static_cast<std::size_t>(stats.moves_total) : 0;
  if (cache_ptr != nullptr) {
    stats.eval_cache_size = static_cast<double>(cache.Size());
    stats.eval_cache_hit_rate =
        cache.Lookups() > 0
            ? static_cast<double>(cache.Hits()) / cache.Lookups()
            : 0.0;
  }
  return stats;
}

int PlayMatch(INetEvaluator &black_evaluator, INetEvaluator &white_evaluator,
              const MctsConfig &mcts_config, int temperature_move_cutoff,
              int max_moves, std::mt19937 &rng) {
  MctsConfig quiet = mcts_config;
  quiet.dirichlet_epsilon_ = 0.0f; // no exploration noise in matches
  Mcts mcts_black, mcts_white;
  Gomoku game;
  std::vector<int> visit_action, visit_count;
  std::array<float, Gomoku::kActionNum> pi;

  while (!game.IsTerminal() && game.move_count() < max_moves) {
    INetEvaluator &evaluator =
        game.current_player() == Gomoku::kBlack ? black_evaluator
                                                : white_evaluator;
    Mcts &mcts =
        game.current_player() == Gomoku::kBlack ? mcts_black : mcts_white;
    mcts.Search(game, quiet, evaluator, rng, visit_action, visit_count);
    Mcts::VisitDistribution(visit_action, visit_count, pi.data());
    int action;
    if (game.move_count() < temperature_move_cutoff) {
      std::discrete_distribution<int> dist(pi.begin(), pi.end());
      action = dist(rng);
    } else {
      action = static_cast<int>(std::max_element(pi.begin(), pi.end()) -
                                pi.begin());
    }
    if (!game.IsLegal(action)) {
      // fall back to the highest-prior legal move
      float best = -1.0f;
      for (int a = 0; a < Gomoku::kActionNum; ++a)
        if (game.IsLegal(a) && pi[a] > best) {
          best = pi[a];
          action = a;
        }
    }
    game.Apply(action);
  }
  return game.IsTerminal() ? game.Result() : 2;
}

void RandomEvaluator::Predict(const Gomoku &game, float *policy,
                              float &value) {
  int legal = 0;
  for (int a = 0; a < Gomoku::kActionNum; ++a)
    if (game.IsLegal(a)) ++legal;
  for (int a = 0; a < Gomoku::kActionNum; ++a)
    policy[a] = game.IsLegal(a) ? 1.0f / legal : 0.0f;
  value = 0.0f;
}

DuelStats RunDuel(deeplearning::PolicyValueResNet &a,
                  deeplearning::PolicyValueResNet &b,
                  const MctsConfig &mcts_config, int game_num, int worker_num,
                  int temperature_move_cutoff, int max_moves, int seed) {
  DuelStats stats;
  std::mutex mutex;
  std::atomic<int> next_game{0};

  auto worker = [&](int worker_index) {
    Evaluator eval_a, eval_b;
    auto config = a.config();
    config.thread_num_ = 1;
    eval_a.Init(config);
    eval_b.Init(config);
    AssignWeights(eval_a.net(), a);
    AssignWeights(eval_b.net(), b);
    std::mt19937 rng(static_cast<unsigned>(seed * 1000003u + worker_index));
    while (true) {
      const int game_index = next_game.fetch_add(1);
      if (game_index >= game_num) break;
      // alternate colors: even games A is black
      INetEvaluator *black = &eval_a, *white = &eval_b;
      bool a_is_black = true;
      if (game_index % 2 == 1) {
        std::swap(black, white);
        a_is_black = false;
      }
      const int result = PlayMatch(*black, *white, mcts_config,
                                   temperature_move_cutoff, max_moves, rng);
      std::lock_guard<std::mutex> lock(mutex);
      if (result == 2) ++stats.draws;
      else if ((result == Gomoku::kBlack) == a_is_black) ++stats.a_wins;
      else ++stats.b_wins;
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < worker_num; ++i) threads.emplace_back(worker, i);
  for (auto &t : threads) t.join();
  return stats;
}

DuelStats RunVsRandom(deeplearning::PolicyValueResNet &net,
                      const MctsConfig &mcts_config, int game_num,
                      int worker_num, int max_moves, int seed) {
  DuelStats stats;
  std::mutex mutex;
  std::atomic<int> next_game{0};
  MctsConfig quiet = mcts_config;
  quiet.dirichlet_epsilon_ = 0.0f;

  auto worker = [&](int worker_index) {
    Evaluator eval;
    auto config = net.config();
    config.thread_num_ = 1;
    eval.Init(config);
    AssignWeights(eval.net(), net);
    Mcts mcts;
    std::vector<int> visit_action, visit_count;
    std::array<float, Gomoku::kActionNum> pi;
    std::mt19937 rng(static_cast<unsigned>(seed * 7919u + worker_index));
    while (true) {
      const int game_index = next_game.fetch_add(1);
      if (game_index >= game_num) break;
      const bool model_black = (game_index % 2 == 0);
      Gomoku game;
      while (!game.IsTerminal() && game.move_count() < max_moves) {
        int action;
        if ((game.current_player() == Gomoku::kBlack) == model_black) {
          mcts.Search(game, quiet, eval, rng, visit_action, visit_count);
          Mcts::VisitDistribution(visit_action, visit_count, pi.data());
          action = static_cast<int>(
              std::max_element(pi.begin(), pi.end()) - pi.begin());
          if (!game.IsLegal(action)) {
            for (action = 0;
                 action < Gomoku::kActionNum && !game.IsLegal(action); ++action) {
            }
          }
        } else {
          // 真随机: 全盘合法手均匀抽
          std::vector<int> legal;
          for (int a = 0; a < Gomoku::kActionNum; ++a)
            if (game.IsLegal(a)) legal.push_back(a);
          std::uniform_int_distribution<int> dist(0,
                                                  static_cast<int>(legal.size()) - 1);
          action = legal[dist(rng)];
        }
        game.Apply(action);
      }
      const int result = game.IsTerminal() ? game.Result() : 2;
      std::lock_guard<std::mutex> lock(mutex);
      if (result == 2) ++stats.draws;
      else if ((result == Gomoku::kBlack) == model_black) ++stats.a_wins;
      else ++stats.b_wins;
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < worker_num; ++i) threads.emplace_back(worker, i);
  for (auto &t : threads) t.join();
  return stats;
}

} // namespace az
