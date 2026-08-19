#pragma once

#include "mcts/mcts.h"
#include "train/evaluator.h"
#include "train/replay_buffer.h"

#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace az {

struct SelfPlayConfig {
  int worker_num_ = 20;          // concurrent game workers (1 net copy each)
  int game_num_ = 40;            // games per iteration
  int max_moves_ = 200;          // force draw beyond this length
  int temperature_move_cutoff_ = 12; // sample from pi before, argmax after
  int seed_ = 42;
  bool use_cache_ = true;
  MctsConfig mcts_;
  // Long-tail acceleration: with probability seed_from_hard_prob_, a game
  // starts from a buffer state where the mover must defend (opponent run
  // threat), instead of an empty board. Data is still 100% self-played.
  float seed_from_hard_prob_ = 0.0f;
  const std::vector<int> *hard_seed_indices_ = nullptr; // buffer indices
};

struct SelfPlayStats {
  int games = 0;
  int black_wins = 0;
  int white_wins = 0;
  int draws = 0;
  std::size_t samples = 0;
  double moves_total = 0.0;
  double eval_cache_size = 0;
  double eval_cache_hit_rate = 0.0;
};

// Runs game_num self-play games with the given master net, adding all
// positions to the buffer. Returns aggregate stats.
SelfPlayStats RunSelfPlay(deeplearning::PolicyValueResNet &master,
                          const SelfPlayConfig &config,
                          ReplayBuffer &buffer);

// Plays one full game between two evaluators starting from a fresh board.
// `black_first_evaluator` plays black. Legacy/default matches force exploration
// noise off; callers may explicitly opt into normalized root noise. Returns
// +1 black wins, -1 white wins, 0 draw (including the move cap).
int PlayMatch(INetEvaluator &black_evaluator, INetEvaluator &white_evaluator,
              const MctsConfig &mcts_config, int temperature_move_cutoff,
              int max_moves, std::mt19937 &rng);

// A player that moves uniformly at random (no search).
class RandomEvaluator : public INetEvaluator {
public:
  void Predict(const Gomoku &game, float *policy, float &value) override;
};

struct DuelStats {
  int a_wins = 0;
  int b_wins = 0;
  int draws = 0;
  int a_black_wins = 0;
  int a_white_wins = 0;
};

// Parallel arena: net A vs net B, colors alternating, no exploration noise,
// greedy play after the temperature cutoff.
DuelStats RunDuel(deeplearning::PolicyValueResNet &a,
                  deeplearning::PolicyValueResNet &b,
                  const MctsConfig &mcts_config, int game_num, int worker_num,
                  int temperature_move_cutoff, int max_moves, int seed);

// Model (MCTS-guided) vs uniform-random player, alternating colors.
// Returns model win/loss/draw counts in stats.a/b/draws.
DuelStats RunVsRandom(deeplearning::PolicyValueResNet &net,
                      const MctsConfig &mcts_config, int game_num,
                      int worker_num, int max_moves, int seed);

} // namespace az
