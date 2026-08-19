#include "arena/gauntlet.h"

#include "arena/js_opponent.h"
#include "train/evaluator.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace az {

namespace {

// One game with the model on `model_black`'s side. Returns model-relative:
// +1 model wins, 0 draw/cap, -1 model loses.
int PlayOneGame(Evaluator &evaluator, JsOpponent &opponent,
                const MctsConfig &mcts_config, bool model_black,
                int max_moves, std::mt19937 &rng) {
  Gomoku game;
  Mcts mcts;
  std::vector<int> visit_action, visit_count;
  std::array<float, Gomoku::kActionNum> pi;

  while (!game.IsTerminal() && game.move_count() < max_moves) {
    const bool model_turn =
        (game.current_player() == Gomoku::kBlack) == model_black;
    int action;
    if (model_turn) {
      mcts.Search(game, mcts_config, evaluator, rng, visit_action,
                  visit_count);
      Mcts::VisitDistribution(visit_action, visit_count, pi.data());
      action = static_cast<int>(
          std::max_element(pi.begin(), pi.end()) - pi.begin());
      if (!game.IsLegal(action)) {
        for (action = 0; action < Gomoku::kActionNum && !game.IsLegal(action);
             ++action) {
        }
      }
    } else {
      action = opponent.PickMove(game);
      if (action < 0) {
        return 1; // opponent crashed / illegal -> model wins by forfeit
      }
    }
    game.Apply(action);
    if (mcts_config.reuse_tree_) mcts.AdvanceRoot(action);
  }
  if (!game.IsTerminal()) {
    return 0;
  }
  const int result = game.Result();
  if (result == 2) {
    return 0;
  }
  const bool black_won = result == Gomoku::kBlack;
  return (black_won == model_black) ? 1 : -1;
}

} // namespace

std::vector<GauntletResult> RunGauntlet(
    deeplearning::PolicyValueResNet &net, const std::vector<int> &levels,
    int games_per_color, int workers, const MctsConfig &mcts_config,
    int max_moves, int seed, GauntletColor color) {
  std::vector<GauntletResult> results;
  std::mutex results_mutex;

  // job queue: (level, model_is_black) pairs
  struct Job {
    int level;
    bool model_black;
    int job_index;
  };
  std::vector<Job> jobs;
  for (int level : levels) {
    for (int i = 0; i < games_per_color; ++i) {
      if (color != GauntletColor::WHITE_ONLY)
        jobs.push_back({level, true, 0});
      if (color != GauntletColor::BLACK_ONLY)
        jobs.push_back({level, false, 0});
    }
  }
  for (std::size_t i = 0; i < jobs.size(); ++i) jobs[i].job_index = (int)i;

  // results indexed by job_index
  std::vector<int> outcomes(jobs.size(), 0);
  std::atomic<int> next_job{0};
  std::atomic<int> finished{0};

  auto worker = [&](int worker_index) {
    Evaluator evaluator;
    auto config = net.config();
    config.thread_num_ = 1;
    evaluator.Init(config);
    AssignWeights(evaluator.net(), net);
    while (true) {
      const int job_index = next_job.fetch_add(1);
      if (job_index >= static_cast<int>(jobs.size())) {
        break;
      }
      const Job &job = jobs[job_index];
      // Per-game seed makes root-noise streams identical across models and
      // independent of worker scheduling.  This matters for fair A/B sweeps.
      std::mt19937 game_rng(
          static_cast<unsigned>(seed * 7919u + job.job_index));
      JsOpponent opponent(job.level);
      if (!opponent.Start()) {
        outcomes[job_index] = 0; // couldn't start: treat as draw, note on log
        std::fprintf(stderr, "[gauntlet] failed to start js engine\n");
      } else {
        outcomes[job_index] = PlayOneGame(evaluator, opponent, mcts_config,
                                          job.model_black, max_moves,
                                          game_rng);
        opponent.Stop();
      }
      const int done = finished.fetch_add(1) + 1;
      std::fprintf(stderr, "\r[gauntlet] %d/%zu", done, jobs.size());
      std::fflush(stderr);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < workers; ++i) threads.emplace_back(worker, i);
  for (auto &t : threads) t.join();
  std::fprintf(stderr, "\n");

  for (std::size_t li = 0; li < levels.size(); ++li) {
    GauntletResult r;
    r.level = levels[li];
    for (std::size_t j = 0; j < jobs.size(); ++j) {
      if (jobs[j].level != levels[li]) continue;
      const int outcome = outcomes[j];
      if (jobs[j].model_black) {
        if (outcome > 0) ++r.black_wins;
        else if (outcome < 0) ++r.black_losses;
        else ++r.black_draws;
      } else {
        if (outcome > 0) ++r.white_wins;
        else if (outcome < 0) ++r.white_losses;
        else ++r.white_draws;
      }
    }
    results.push_back(r);
  }
  return results;
}

} // namespace az
