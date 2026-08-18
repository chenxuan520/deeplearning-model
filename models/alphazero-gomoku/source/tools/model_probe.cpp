#include "cnn/policy_value_resnet.h"
#include "game/gomoku.h"
#include "mcts/mcts.h"
#include "train/evaluator.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: az_model_probe MODEL [action0,action1,...] [sims]\n"
               "actions are row-major cell ids (0..224)\n");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    Usage();
    return 1;
  }
  deeplearning::PolicyValueResNet net;
  if (net.Load(argv[1]) != deeplearning::PolicyValueResNet::SUCCESS) {
    std::fprintf(stderr, "load failed: %s\n", net.err_msg().c_str());
    return 1;
  }
  az::Gomoku game;
  if (argc >= 3 && argv[2][0] != '\0') {
    std::stringstream input(argv[2]);
    std::string token;
    while (std::getline(input, token, ',')) {
      const int action = std::atoi(token.c_str());
      if (!game.Apply(action)) {
        std::fprintf(stderr, "invalid action: %d\n", action);
        return 1;
      }
    }
  }
  deeplearning::FloatTensor4D encoded;
  game.Encode(encoded);
  deeplearning::PolicyValueResNet::Output output;
  if (net.Forward(encoded, output, false) !=
      deeplearning::PolicyValueResNet::SUCCESS) {
    std::fprintf(stderr, "forward failed: %s\n", net.err_msg().c_str());
    return 1;
  }
  int mcts_action = -1;
  if (argc >= 4) {
    const int sims = std::atoi(argv[3]);
    az::Evaluator evaluator;
    auto config = net.config();
    config.thread_num_ = 1;
    if (!evaluator.Init(config) || !az::AssignWeights(evaluator.net(), net)) {
      std::fprintf(stderr, "evaluator init failed\n");
      return 1;
    }
    az::Mcts mcts;
    az::MctsConfig mcts_config;
    mcts_config.simulation_num_ = sims;
    mcts_config.dirichlet_epsilon_ = 0.0f;
    std::mt19937 rng(20260816);
    std::vector<int> actions, visits;
    mcts.Search(game, mcts_config, evaluator, rng, actions, visits);
    int best_visits = -1;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (visits[i] > best_visits) {
        best_visits = visits[i];
        mcts_action = actions[i];
      }
    }
  }
  std::printf("{\"current_player\":%d,\"last_action\":%d,"
              "\"value\":%.9g,\"mcts_action\":%d,\"logits\":[",
              game.current_player(), game.last_action(), output.values_[0],
              mcts_action);
  for (int i = 0; i < az::Gomoku::kActionNum; ++i) {
    if (i != 0) std::putchar(',');
    std::printf("%.9g", output.policy_logits_[i]);
  }
  std::printf("]}\n");
  return 0;
}
