// alphazero-gomoku CLI: bench / train / eval / arena / play / info
//
// All training runs on the framework components under lib/ (PolicyValueResNet
// + PolicyValueLoss + FloatAdamW + ThreadPool).

#include "cnn/policy_value_resnet.h"
#include "arena/gauntlet.h"
#include "game/gomoku.h"
#include "serve/serve.h"
#include "train/trainer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using az::Gomoku;
using deeplearning::FloatTensor4D;
using deeplearning::PolicyValueResNet;

deeplearning::PolicyValueResNet::Config MakeNetConfig(int trunk, int blocks,
                                                      int thread_num,
                                                      int seed) {
  deeplearning::PolicyValueResNet::Config config;
  config.input_channels_ = Gomoku::kPlaneNum;
  config.board_height_ = Gomoku::kBoardSize;
  config.board_width_ = Gomoku::kBoardSize;
  config.trunk_channels_ = trunk;
  config.residual_block_num_ = blocks;
  config.policy_channels_ = 2;
  config.policy_size_ = Gomoku::kActionNum;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 64;
  config.thread_num_ = thread_num;
  config.rand_seed_ = seed;
  return config;
}

void PrintUsage() {
  std::printf(
      "usage: alphazero <command> [options]\n"
      "  train  (AlphaZero loop: self-play + train + gate; see options below)\n"
      "  eval   --model FILE [--games 20] [--sims 100] [--workers 8] [--reuse-tree 0]\n"
      "  arena  --model-a FILE --model-b FILE [--games 20] [--sims 100]\n"
      "         [--workers 8] [--temp-moves 0] [--dir-eps 0] [--dir-alpha .3]\n"
      "         [--deterministic-games 0] [--reuse-tree 0]\n"
      "  play   [--model FILE] [--sims 100] [--reuse-tree 0] (you play white)\n"
      "  serve  [--model FILE] [--port 8765] [--sims 800] [--threads 24]\n"
      "           (HTTP move API for the web frontend; CORS open)\n"
      "  gauntlet [--model FILE] [--games 10] [--workers 8] [--levels 6,7]\n"
      "           [--color both|black|white] [--dir-eps 0] [--dir-alpha .3] [--reuse-tree 0]\n"
      "           (acceptance vs game-old JS levels 1-7)\n"
      "  bench  [--trunk N] [--blocks N] [--batch B] [--threads T] [--iters N]\n"
      "  bench  --concurrent N [--iters N]\n"
      "  info\n"
      "train options:\n"
      "  --run-dir DIR (runtime)  --iterations N (-1 forever)  --seed N\n"
      "  --workers N (20)  --games-per-iter N (40)  --sims N (100)\n"
      "  --train-steps N (150)  --batch N (256)\n"
      "  --lr X (1e-3)  --wd X (1e-4)  --buffer N (200000)\n"
      "  --max-moves N (200)  --temp-moves N (12)\n"
      "  --cpuct X (1.5)  --dir-eps X (0.25)  --dir-alpha X (0.3)\n"
      "  --reuse-tree 0|1 (0)\n"
      "  --trunk N (32)  --blocks N (4)\n"
      "  --gate-every N (5)  --gate-games N (20)  --gate-threshold X (0.55)\n"
      "  --no-resume  --no-cache  --save-buffer-every N (10)\n");
}

// ---------------- bench ----------------

int CmdBench(int argc, char **argv) {
  int trunk = 32, blocks = 4, batch = 1, threads = 4, iters = 20,
      concurrent = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--trunk") == 0) trunk = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--blocks") == 0) blocks = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--batch") == 0) batch = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--threads") == 0) threads = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--concurrent") == 0)
      concurrent = std::atoi(argv[i + 1]);
  }

  if (concurrent > 0) {
    std::vector<std::unique_ptr<PolicyValueResNet>> nets(concurrent);
    std::vector<std::unique_ptr<FloatTensor4D>> inputs(concurrent);
    std::vector<std::unique_ptr<PolicyValueResNet::Output>> outputs(
        concurrent);
    std::atomic<int> failures{0};
    auto config = MakeNetConfig(trunk, blocks, 1, 42);
    for (int i = 0; i < concurrent; ++i) {
      nets[i].reset(new PolicyValueResNet());
      config.rand_seed_ = 42 + i;
      if (nets[i]->Init(config) != PolicyValueResNet::SUCCESS) {
        std::printf("net %d init failed\n", i);
        return 1;
      }
      inputs[i].reset(new FloatTensor4D());
      inputs[i]->Resize(1, Gomoku::kPlaneNum, Gomoku::kBoardSize,
                        Gomoku::kBoardSize, 0.0f);
      outputs[i].reset(new PolicyValueResNet::Output());
    }
    std::printf("concurrent=%d single-thread nets, iters=%d each\n",
                concurrent, iters);
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < concurrent; ++i) {
      pool.emplace_back([&, i]() {
        for (int k = 0; k < iters; ++k) {
          if (nets[i]->Forward(*inputs[i], *outputs[i], false) !=
              PolicyValueResNet::SUCCESS) {
            ++failures;
          }
        }
      });
    }
    for (auto &t : pool) t.join();
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    std::printf("aggregate: %.1f evals/s (%.2f ms/eval per net)\n",
                concurrent * iters / seconds, seconds * 1000.0 / iters);
    return failures == 0 ? 0 : 1;
  }

  PolicyValueResNet net;
  auto config = MakeNetConfig(trunk, blocks, threads, 42);
  if (net.Init(config) != PolicyValueResNet::SUCCESS) {
    std::printf("net init failed: %s\n", net.err_msg().c_str());
    return 1;
  }
  std::printf("net: trunk=%d blocks=%d params=%zu batch=%d threads=%d\n",
              trunk, blocks, net.parameter_count(), batch, threads);

  FloatTensor4D input;
  input.Resize(batch, Gomoku::kPlaneNum, Gomoku::kBoardSize,
               Gomoku::kBoardSize, 0.0f);
  for (int i = 0; i < 30 && i < Gomoku::kCellNum; ++i) {
    input.values()[i * 7 + 3] = 1.0f;
  }
  PolicyValueResNet::Output output;
  for (int i = 0; i < 3; ++i) net.Forward(input, output, false);
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (net.Forward(input, output, false) != PolicyValueResNet::SUCCESS) {
      std::printf("forward failed: %s\n", net.err_msg().c_str());
      return 1;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(end - begin).count() / iters;
  std::printf("forward: %.2f ms/call (%.1f evals/s, value[0]=%.4f)\n", ms,
              batch * 1000.0 / ms,
              output.values_.empty() ? 0.0f : output.values_[0]);
  return 0;
}

// ---------------- shared helpers ----------------

bool LoadNet(const std::string &path, PolicyValueResNet &net) {
  if (net.Load(path) != PolicyValueResNet::SUCCESS) {
    std::printf("load %s failed: %s\n", path.c_str(),
                net.err_msg().c_str());
    return false;
  }
  return true;
}

const char *ResultText(const az::DuelStats &s) {
  static char buf[128];
  std::snprintf(buf, sizeof(buf), "wins=%d losses=%d draws=%d", s.a_wins,
                s.b_wins, s.draws);
  return buf;
}

// ---------------- eval / arena ----------------

int CmdEval(int argc, char **argv) {
  std::string model = "runtime/best.net";
  int games = 20, sims = 100, workers = 8, max_moves = 200, seed = 7;
  bool reuse_tree = false;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--model")) model = argv[i + 1];
    if (!std::strcmp(argv[i], "--games")) games = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--sims")) sims = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--workers")) workers = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--max-moves"))
      max_moves = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--reuse-tree"))
      reuse_tree = std::atoi(argv[i + 1]) != 0;
  }
  PolicyValueResNet net;
  if (!LoadNet(model, net)) return 1;
  az::MctsConfig mcts;
  mcts.simulation_num_ = sims;
  mcts.dirichlet_epsilon_ = 0.0f;
  mcts.reuse_tree_ = reuse_tree;
  auto stats = az::RunVsRandom(net, mcts, games, workers, max_moves, seed);
  std::printf("vs random (model alternates colors): %s  win-rate=%.1f%%\n",
              ResultText(stats),
              100.0 * stats.a_wins / std::max(1, games));
  return 0;
}

int CmdArena(int argc, char **argv) {
  std::string model_a = "runtime/latest.net", model_b = "runtime/best.net";
  int games = 20, sims = 100, workers = 8, max_moves = 200, seed = 7;
  int temp_moves = 0;
  float dir_eps = 0.0f, dir_alpha = 0.3f;
  bool deterministic_games = false, reuse_tree = false;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--model-a")) model_a = argv[i + 1];
    if (!std::strcmp(argv[i], "--model-b")) model_b = argv[i + 1];
    if (!std::strcmp(argv[i], "--games")) games = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--sims")) sims = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--workers")) workers = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--max-moves"))
      max_moves = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--temp-moves"))
      temp_moves = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--dir-eps"))
      dir_eps = static_cast<float>(std::atof(argv[i + 1]));
    if (!std::strcmp(argv[i], "--dir-alpha"))
      dir_alpha = static_cast<float>(std::atof(argv[i + 1]));
    if (!std::strcmp(argv[i], "--deterministic-games"))
      deterministic_games = std::atoi(argv[i + 1]) != 0;
    if (!std::strcmp(argv[i], "--reuse-tree"))
      reuse_tree = std::atoi(argv[i + 1]) != 0;
  }
  PolicyValueResNet a, b;
  if (!LoadNet(model_a, a) || !LoadNet(model_b, b)) return 1;
  az::MctsConfig mcts;
  mcts.simulation_num_ = sims;
  mcts.dirichlet_epsilon_ = dir_eps;
  mcts.dirichlet_alpha_ = dir_alpha;
  mcts.normalized_dirichlet_ = dir_eps > 0.0f;
  mcts.deterministic_game_seeds_ = deterministic_games;
  mcts.reuse_tree_ = reuse_tree;
  auto stats =
      az::RunDuel(a, b, mcts, games, workers, temp_moves, max_moves, seed);
  const int decisive = stats.a_wins + stats.b_wins;
  std::printf("arena %s vs %s: A %s  (rate=%.2f over %d decisive)\n",
              model_a.c_str(), model_b.c_str(), ResultText(stats),
              decisive > 0 ? static_cast<double>(stats.a_wins) / decisive
                            : 0.5,
              decisive);
  std::printf("A color split: black_wins=%d white_wins=%d\n",
              stats.a_black_wins, stats.a_white_wins);
  return 0;
}

// ---------------- gauntlet (acceptance vs game-old levels 1-7) ----------------

int CmdGauntlet(int argc, char **argv) {
  std::string model = "runtime/best.net";
  std::vector<int> levels = {1, 2, 3, 4, 5, 6, 7};
  int games = 10, workers = 8, sims = 100, max_moves = 225, seed = 99;
  float dir_eps = 0.0f, dir_alpha = 0.3f;
  bool reuse_tree = false;
  az::GauntletColor color = az::GauntletColor::BOTH;
  const char *color_name = "both";
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--model")) model = argv[i + 1];
    if (!std::strcmp(argv[i], "--games")) games = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--workers")) workers = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--sims")) sims = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--max-moves"))
      max_moves = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--dir-eps"))
      dir_eps = static_cast<float>(std::atof(argv[i + 1]));
    if (!std::strcmp(argv[i], "--dir-alpha"))
      dir_alpha = static_cast<float>(std::atof(argv[i + 1]));
    if (!std::strcmp(argv[i], "--reuse-tree"))
      reuse_tree = std::atoi(argv[i + 1]) != 0;
    if (!std::strcmp(argv[i], "--color")) {
      color_name = argv[i + 1];
      if (!std::strcmp(color_name, "black"))
        color = az::GauntletColor::BLACK_ONLY;
      else if (!std::strcmp(color_name, "white"))
        color = az::GauntletColor::WHITE_ONLY;
      else if (!std::strcmp(color_name, "both"))
        color = az::GauntletColor::BOTH;
      else {
        std::fprintf(stderr, "invalid --color (expected both|black|white)\n");
        return 1;
      }
    }
    if (!std::strcmp(argv[i], "--levels")) {
      levels.clear();
      std::string spec = argv[i + 1];
      std::size_t begin = 0;
      while (begin <= spec.size()) {
        const std::size_t comma = spec.find(',', begin);
        levels.push_back(std::atoi(spec.substr(begin, comma == std::string::npos
                                                          ? std::string::npos
                                                          : comma - begin)
                                       .c_str()));
        if (comma == std::string::npos) break;
        begin = comma + 1;
      }
    }
  }
  deeplearning::PolicyValueResNet net;
  if (!LoadNet(model, net)) {
    return 1;
  }
  az::MctsConfig mcts;
  mcts.simulation_num_ = sims;
  mcts.dirichlet_epsilon_ = dir_eps;
  mcts.dirichlet_alpha_ = dir_alpha;
  mcts.normalized_dirichlet_ = dir_eps > 0.0f;
  mcts.reuse_tree_ = reuse_tree;
  std::printf("gauntlet: %s vs %zu levels x %d games, color=%s\n",
              model.c_str(), levels.size(), games, color_name);
  auto results = az::RunGauntlet(net, levels, games, workers, mcts, max_moves,
                                 seed, color);
  std::printf("\nlevel | black W/L/D (rate) | white W/L/D (rate)\n");
  for (const auto &r : results) {
    const int black_games = r.black_wins + r.black_losses + r.black_draws;
    const int white_games = r.white_wins + r.white_losses + r.white_draws;
    std::printf(
        "%5d | %5d/%-2d/%d (%5.1f%%) | %5d/%-2d/%d (%5.1f%%)\n", r.level,
        r.black_wins, r.black_losses, r.black_draws,
        100.0 * r.black_wins / std::max(1, black_games), r.white_wins,
        r.white_losses, r.white_draws,
        100.0 * r.white_wins / std::max(1, white_games));
  }
  return 0;
}

// ---------------- play ----------------

void PrintBoard(const Gomoku &game) {
  const auto &board = game.board();
  std::printf("   ");
  for (int c = 0; c < Gomoku::kBoardSize; ++c) std::printf("%2d ", c);
  std::printf("\n");
  for (int r = 0; r < Gomoku::kBoardSize; ++r) {
    std::printf("%2d ", r);
    for (int c = 0; c < Gomoku::kBoardSize; ++c) {
      const int cell = r * Gomoku::kBoardSize + c;
      char ch = board[cell] == Gomoku::kBlack
                    ? 'X'
                    : board[cell] == Gomoku::kWhite ? 'O' : '.';
      if (cell == game.last_action()) {
        std::printf("[%c]", ch);
      } else {
        std::printf(" %c ", ch);
      }
    }
    std::printf("\n");
  }
}

int CmdPlay(int argc, char **argv) {
  std::string model = "runtime/best.net";
  int sims = 100;
  bool reuse_tree = false;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--model")) model = argv[i + 1];
    if (!std::strcmp(argv[i], "--sims")) sims = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--reuse-tree"))
      reuse_tree = std::atoi(argv[i + 1]) != 0;
  }
  PolicyValueResNet net;
  if (!LoadNet(model, net)) return 1;
  az::Evaluator evaluator;
  auto config = net.config();
  config.thread_num_ = 4;
  evaluator.Init(config);
  az::AssignWeights(evaluator.net(), net);

  az::MctsConfig mcts;
  mcts.simulation_num_ = sims;
  mcts.dirichlet_epsilon_ = 0.0f;
  mcts.reuse_tree_ = reuse_tree;
  az::Mcts search;
  std::mt19937 rng(1234);

  Gomoku game;
  std::printf("you are white (O). enter moves as \"row col\". X = model.\n");
  PrintBoard(game);
  std::vector<int> visit_action, visit_count;
  std::array<float, Gomoku::kActionNum> pi;
  while (!game.IsTerminal()) {
    int action;
    if (game.current_player() == Gomoku::kBlack) {
      search.Search(game, mcts, evaluator, rng, visit_action, visit_count);
      az::Mcts::VisitDistribution(visit_action, visit_count, pi.data());
      action = static_cast<int>(std::max_element(pi.begin(), pi.end()) -
                                pi.begin());
      int row, column;
      Gomoku::CellXY(action, row, column);
      std::printf("model plays %d %d\n", row, column);
    } else {
      std::printf("your move> ");
      std::fflush(stdout);
      int row, column;
      if (std::scanf("%d %d", &row, &column) != 2) {
        break;
      }
      action = row * Gomoku::kBoardSize + column;
      if (!game.IsLegal(action)) {
        std::printf("illegal move\n");
        continue;
      }
    }
    game.Apply(action);
    if (mcts.reuse_tree_) search.AdvanceRoot(action);
    PrintBoard(game);
  }
  const int result = game.Result();
  std::printf("%s\n", result == Gomoku::kBlack
                          ? "model (black) wins"
                          : result == Gomoku::kWhite ? "you win" : "draw");
  return 0;
}

// ---------------- info ----------------

int CmdInfo() {
  PolicyValueResNet net;
  if (net.Init(MakeNetConfig(32, 4, 1, 42)) != PolicyValueResNet::SUCCESS) {
    std::printf("net init failed: %s\n", net.err_msg().c_str());
    return 1;
  }
  std::printf("default: trunk=32 blocks=4 planes=%d policy=%d params=%zu "
              "(%.2f M)\n",
              Gomoku::kPlaneNum, Gomoku::kActionNum, net.parameter_count(),
              net.parameter_count() / 1e6);
  return 0;
}

// ---------------- train ----------------

int CmdTrain(int argc, char **argv) {
  az::TrainConfig config;
  // sensible defaults for 15x15 CPU training
  config.net_ = MakeNetConfig(32, 4, 1, config.seed_);
  config.selfplay_.worker_num_ = 20;
  config.selfplay_.game_num_ = 40;
  config.selfplay_.max_moves_ = 200;
  config.selfplay_.temperature_move_cutoff_ = 12;
  config.selfplay_.mcts_.simulation_num_ = 100;

  for (int i = 2; i + 1 < argc; i += 2) {
    const char *key = argv[i];
    const char *value = argv[i + 1];
    if (!std::strcmp(key, "--run-dir")) config.run_dir_ = value;
    else if (!std::strcmp(key, "--iterations")) config.iterations_ = std::atoi(value);
    else if (!std::strcmp(key, "--seed")) config.seed_ = std::atoi(value);
    else if (!std::strcmp(key, "--workers")) config.selfplay_.worker_num_ = std::atoi(value);
    else if (!std::strcmp(key, "--games-per-iter")) config.selfplay_.game_num_ = std::atoi(value);
    else if (!std::strcmp(key, "--sims")) config.selfplay_.mcts_.simulation_num_ = std::atoi(value);
    else if (!std::strcmp(key, "--train-steps")) config.train_steps_ = std::atoi(value);
    else if (!std::strcmp(key, "--batch")) config.batch_size_ = std::atoi(value);
    else if (!std::strcmp(key, "--lr")) config.learning_rate_ = std::atof(value);
    else if (!std::strcmp(key, "--wd")) config.weight_decay_ = std::atof(value);
    else if (!std::strcmp(key, "--value-weight")) config.value_weight_ = std::atof(value);
    else if (!std::strcmp(key, "--buffer")) config.buffer_capacity_ = std::stoul(value);
    else if (!std::strcmp(key, "--max-moves")) config.selfplay_.max_moves_ = std::atoi(value);
    else if (!std::strcmp(key, "--temp-moves")) config.selfplay_.temperature_move_cutoff_ = std::atoi(value);
    else if (!std::strcmp(key, "--seed-hard-prob")) config.selfplay_.seed_from_hard_prob_ = std::atof(value);
    else if (!std::strcmp(key, "--cpuct")) config.selfplay_.mcts_.c_puct_ = std::atof(value);
    else if (!std::strcmp(key, "--dir-eps")) config.selfplay_.mcts_.dirichlet_epsilon_ = std::atof(value);
    else if (!std::strcmp(key, "--dir-alpha")) config.selfplay_.mcts_.dirichlet_alpha_ = std::atof(value);
    else if (!std::strcmp(key, "--fpu")) config.selfplay_.mcts_.fpu_reduction_ = std::atof(value);
    else if (!std::strcmp(key, "--reuse-tree")) config.selfplay_.mcts_.reuse_tree_ = std::atoi(value) != 0;
    else if (!std::strcmp(key, "--gate-every")) config.gate_every_ = std::atoi(value);
    else if (!std::strcmp(key, "--gate-games")) config.gate_games_ = std::atoi(value);
    else if (!std::strcmp(key, "--gate-threshold")) config.gate_threshold_ = std::atof(value);
    else if (!std::strcmp(key, "--save-buffer-every")) config.save_buffer_every_ = std::atoi(value);
    else if (!std::strcmp(key, "--trunk")) config.net_.trunk_channels_ = std::atoi(value);
    else if (!std::strcmp(key, "--blocks")) config.net_.residual_block_num_ = std::atoi(value);
    else {
      std::printf("unknown option: %s\n", key);
      return 1;
    }
  }
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--no-resume")) config.resume_ = false;
    if (!std::strcmp(argv[i], "--no-cache")) config.selfplay_.use_cache_ = false;
  }
  config.selfplay_.seed_ = config.seed_;

  az::Trainer trainer;
  if (!trainer.Init(config)) {
    return 1;
  }
  std::printf("training starts: workers=%d games/iter=%d sims=%d steps=%d "
              "batch=%d lr=%.4g run-dir=%s\n",
              config.selfplay_.worker_num_, config.selfplay_.game_num_,
              config.selfplay_.mcts_.simulation_num_, config.train_steps_,
              config.batch_size_, config.learning_rate_,
              config.run_dir_.c_str());
  trainer.Run();
  return 0;
}

// ---------------- serve (HTTP move API) ----------------

int CmdServe(int argc, char **argv) {
  std::string model = "runtime/champion_iter243.net";
  int port = 8765, sims = 800, threads = 24;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--model")) model = argv[i + 1];
    if (!std::strcmp(argv[i], "--port")) port = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--sims")) sims = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(argv[i + 1]);
  }
  return az::ServeMoves(model, port, sims, threads);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }
  const std::string command = argv[1];
  if (command == "bench") return CmdBench(argc, argv);
  if (command == "train") return CmdTrain(argc, argv);
  if (command == "eval") return CmdEval(argc, argv);
  if (command == "arena") return CmdArena(argc, argv);
  if (command == "gauntlet") return CmdGauntlet(argc, argv);
  if (command == "play") return CmdPlay(argc, argv);
  if (command == "info") return CmdInfo();
  if (command == "serve") return CmdServe(argc, argv);
  PrintUsage();
  return 1;
}
