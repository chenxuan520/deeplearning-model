#pragma once

#include "cnn/policy_value_resnet.h"
#include "mcts/mcts.h"

#include <vector>

namespace az {

struct GauntletResult {
  int level = 0;
  // model playing black (first) / white (second)
  int black_wins = 0, black_losses = 0, black_draws = 0;
  int white_wins = 0, white_losses = 0, white_draws = 0;
};

enum class GauntletColor {
  BOTH,
  BLACK_ONLY,
  WHITE_ONLY,
};

// Plays games_for_each_color x2 games vs each requested JS level, model
// alternating colors. workers = parallel games, each with its own node
// subprocess + net copy.
std::vector<GauntletResult> RunGauntlet(
    deeplearning::PolicyValueResNet &net, const std::vector<int> &levels,
    int games_per_color, int workers, const MctsConfig &mcts_config,
    int max_moves, int seed, GauntletColor color = GauntletColor::BOTH);

} // namespace az
