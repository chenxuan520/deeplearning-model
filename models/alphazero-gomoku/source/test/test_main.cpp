// Minimal test harness: CHECK records failures and keeps going.

#include "game/gomoku.h"
#include "mcts/mcts.h"
#include "train/evaluator.h"
#include "train/replay_buffer.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                        \
  } while (0)

namespace {

using az::Gomoku;

// Plays actions in order; returns false early if any move was illegal.
bool PlayMoves(Gomoku &game, std::initializer_list<int> actions) {
  for (int action : actions) {
    if (!game.Apply(action)) {
      return false;
    }
  }
  return true;
}

int A(int row, int column) { return row * Gomoku::kBoardSize + column; }

void TestHorizontalWin() {
  Gomoku game;
  // Black: (7,3)..(7,7), White plays elsewhere.
  CHECK(PlayMoves(game, {A(7, 3), A(0, 0), A(7, 4), A(0, 1), A(7, 5), A(0, 2),
                         A(7, 6), A(0, 3), A(7, 7)}));
  CHECK(game.IsTerminal());
  CHECK(game.Result() == Gomoku::kBlack);
  CHECK(game.current_player() == Gomoku::kBlack); // no switch after terminal
}

void TestVerticalAndDiagonalWin() {
  {
    Gomoku game;
    CHECK(PlayMoves(game, {A(2, 5), A(13, 0), A(3, 5), A(13, 1), A(4, 5),
                           A(13, 2), A(5, 5), A(13, 3), A(6, 5)}));
    CHECK(game.Result() == Gomoku::kBlack);
  }
  {
    Gomoku game;
    // Black diagonal (1,1)(2,2)(3,3)(4,4)(5,5); white on row 13.
    CHECK(PlayMoves(game, {A(1, 1), A(13, 0), A(2, 2), A(13, 1), A(3, 3),
                           A(13, 2), A(4, 4), A(13, 3), A(5, 5)}));
    CHECK(game.Result() == Gomoku::kBlack);
  }
  {
    Gomoku game;
    // Black anti-diagonal: (1,10)(2,9)(3,8)(4,7)(5,6); white on row 13.
    CHECK(PlayMoves(game, {A(1, 10), A(13, 0), A(2, 9), A(13, 1), A(3, 8),
                           A(13, 2), A(4, 7), A(13, 3), A(5, 6)}));
    CHECK(game.Result() == Gomoku::kBlack);
  }
}

void TestIllegalMoves() {
  Gomoku game;
  CHECK(game.Apply(A(7, 7)));
  CHECK(!game.Apply(A(7, 7)));       // occupied
  CHECK(!game.Apply(-1));            // out of range
  CHECK(!game.Apply(Gomoku::kCellNum)); // out of range
  CHECK(game.move_count() == 1);
}

void TestDraw() {
  Gomoku game;
  // Full-board pattern with no five-in-a-row for either color and exactly
  // 113 black / 112 white cells so a fully alternating playout fits.
  // Row cell (r,c) is black iff ((c + 2r + s_r) mod 8) < 4 with the
  // per-row phase offsets below (found by exhaustive backtracking over
  // phases; no_five below re-verifies the result independently).
  static const int kRowShift[15] = {0, 6, 4, 4, 6, 6, 4, 4,
                                    6, 7, 4, 6, 1, 7, 5};
  int black_actions[Gomoku::kCellNum];
  int white_actions[Gomoku::kCellNum];
  int black_num = 0, white_num = 0;
  for (int row = 0; row < Gomoku::kBoardSize; ++row) {
    const int shift = kRowShift[row];
    for (int column = 0; column < Gomoku::kBoardSize; ++column) {
      const bool is_black = ((column + 2 * row + shift) % 8) < 4;
      if (is_black) {
        black_actions[black_num++] = A(row, column);
      } else {
        white_actions[white_num++] = A(row, column);
      }
    }
  }
  // Check pattern truly has no 5-in-a-row before playing it out.
  auto no_five = [](const int *cells, int num) {
    std::array<int8_t, Gomoku::kCellNum> filled{};
    for (int i = 0; i < num; ++i) filled[cells[i]] = 1;
    static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int r = 0; r < Gomoku::kBoardSize; ++r) {
      for (int c = 0; c < Gomoku::kBoardSize; ++c) {
        if (!filled[r * Gomoku::kBoardSize + c]) continue;
        for (const auto &d : kDirs) {
          int count = 1;
          for (int s = 1; s < 5; ++s) {
            const int rr = r + d[0] * s, cc = c + d[1] * s;
            if (rr < 0 || rr >= Gomoku::kBoardSize || cc < 0 ||
                cc >= Gomoku::kBoardSize ||
                !filled[rr * Gomoku::kBoardSize + cc])
              break;
            ++count;
          }
          if (count >= 5) return false;
        }
      }
    }
    return true;
  };
  CHECK(no_five(black_actions, black_num));
  CHECK(no_five(white_actions, white_num));
  CHECK(black_num == (Gomoku::kCellNum + 1) / 2);
  CHECK(white_num == Gomoku::kCellNum / 2);

  // Interleave moves by move number: black first (113 blacks, 112 whites).
  bool ok = true;
  for (int move = 0; move < Gomoku::kCellNum; ++move) {
    const int action = (move % 2 == 0) ? black_actions[move / 2]
                                       : white_actions[move / 2];
    if (!game.Apply(action)) {
      std::printf("draw playout broke at move %d action %d\n", move, action);
      ok = false;
      break;
    }
  }
  CHECK(ok);
  CHECK(game.move_count() == Gomoku::kCellNum);
  CHECK(game.Result() == 2); // draw
}

void TestEncode() {
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)})); // black, then white
  deeplearning::FloatTensor4D tensor;
  game.Encode(tensor);
  CHECK(tensor.batch() == 1 && tensor.channels() == Gomoku::kPlaneNum);
  CHECK(tensor.height() == 15 && tensor.width() == 15);
  // Black to move: own = black stones, opponent = white stones.
  CHECK(tensor(0, 0, 7, 7) == 1.0f); // own stone
  CHECK(tensor(0, 1, 8, 8) == 1.0f); // opponent stone
  CHECK(tensor(0, 2, 8, 8) == 1.0f); // last move marker
  float side_sum = 0.0f;
  for (int i = 0; i < Gomoku::kCellNum; ++i) {
    side_sum += tensor.values()[3 * Gomoku::kCellNum + i];
  }
  CHECK(side_sum == static_cast<float>(Gomoku::kCellNum)); // black to move
  float own_sum = 0.0f;
  for (int i = 0; i < Gomoku::kCellNum; ++i) {
    own_sum += tensor.values()[i];
  }
  CHECK(own_sum == 1.0f);

  Gomoku restored;
  restored.MutableBoard()[A(7, 7)] = Gomoku::kBlack;
  restored.MutableBoard()[A(8, 8)] = Gomoku::kWhite;
  restored.SetState(Gomoku::kBlack, 2, 0, A(8, 8));
  restored.Encode(tensor);
  CHECK(tensor(0, 2, 8, 8) == 1.0f);
}

void TestEvalCacheIncludesLastMove() {
  Gomoku first;
  first.MutableBoard()[A(7, 7)] = Gomoku::kBlack;
  first.MutableBoard()[A(8, 8)] = Gomoku::kWhite;
  first.SetState(Gomoku::kBlack, 2, 0, A(7, 7));
  Gomoku second = first;
  second.SetState(Gomoku::kBlack, 2, 0, A(8, 8));

  az::EvalCache cache;
  float policy[Gomoku::kActionNum] = {};
  policy[A(6, 6)] = 1.0f;
  cache.Store(first, policy, 0.25f);
  az::EvalCache::Entry entry;
  CHECK(cache.Lookup(first, entry));
  CHECK(std::fabs(entry.value - 0.25f) < 1e-6f);
  CHECK(!cache.Lookup(second, entry));
}

void TestSymmetryConsistency() {
  using az::Gomoku;
  // All 8 transforms are bijections on cells.
  for (int s = 0; s < Gomoku::kSymmetryNum; ++s) {
    std::array<bool, Gomoku::kCellNum> seen{};
    for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
      const int mapped = Gomoku::TransformAction(cell, s);
      CHECK(mapped >= 0 && mapped < Gomoku::kCellNum);
      seen[mapped] = true;
    }
    for (bool found : seen) CHECK(found);
  }
  // Transform twice with identity-symmetric op pairs recovers original:
  // rotation 180 twice = identity (s=4 -> rotate180 twice).
  for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
    CHECK(Gomoku::TransformAction(Gomoku::TransformAction(cell, 4), 4) ==
          cell);
    CHECK(Gomoku::TransformAction(Gomoku::TransformAction(cell, 1), 1) ==
          cell); // mirror twice = identity
  }
  // Planes/policies transform identically with a delta input.
  Gomoku game;
  CHECK(PlayMoves(game, {A(2, 3), A(5, 6), A(9, 1)}));
  deeplearning::FloatTensor4D tensor;
  game.Encode(tensor);
  for (int s = 0; s < Gomoku::kSymmetryNum; ++s) {
    std::vector<float> planes_out(Gomoku::kPlaneNum * Gomoku::kCellNum);
    Gomoku::TransformPlanes(tensor.data(), planes_out.data(), s);
    // own-stone count must be preserved under any symmetry.
    float own_sum = 0.0f;
    for (int i = 0; i < Gomoku::kCellNum; ++i) own_sum += planes_out[i];
    float own_expected = 0.0f;
    for (int i = 0; i < Gomoku::kCellNum; ++i) own_expected += tensor.values()[i];
    CHECK(std::fabs(own_sum - own_expected) < 1e-4f);
  }
}

} // namespace

// Uniform-prior / zero-value fake evaluator for MCTS mechanics tests.
class FlatEvaluator : public az::INetEvaluator {
public:
  void Predict(const az::Gomoku &game, float *policy, float &value) override {
    int legal = 0;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      if (game.IsLegal(a)) ++legal;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      policy[a] = game.IsLegal(a) ? 1.0f / legal : 0.0f;
    value = 0.0f;
  }
};

class CountingEvaluator : public az::INetEvaluator {
public:
  int calls = 0;
  void Predict(const az::Gomoku &game, float *policy, float &value) override {
    ++calls;
    int legal = 0;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      if (game.IsLegal(a)) ++legal;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      policy[a] = game.IsLegal(a) ? 1.0f / legal : 0.0f;
    value = 0.0f;
  }
};

namespace {

void TestMctsForcedWin() {
  using az::Gomoku;
  // Black has an open four at (7,3..6); winning replies: (7,2) and (7,7).
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 3), A(0, 0), A(7, 4), A(0, 1), A(7, 5), A(0, 2),
                         A(7, 6), A(0, 3)}));
  CHECK(!game.IsTerminal());
  CHECK(game.current_player() == Gomoku::kBlack);

  FlatEvaluator evaluator;
  az::MctsConfig config;
  config.simulation_num_ = 300;
  config.dirichlet_epsilon_ = 0.0f;
  az::Mcts mcts;
  std::mt19937 rng(1);
  std::vector<int> visit_action, visit_count;
  mcts.Search(game, config, evaluator, rng, visit_action, visit_count);

  int best = -1, best_count = -1;
  for (std::size_t i = 0; i < visit_action.size(); ++i) {
    if (visit_count[i] > best_count) {
      best_count = visit_count[i];
      best = visit_action[i];
    }
  }
  // MCTS must have found the immediate win.
  CHECK(best == A(7, 2) || best == A(7, 7));
}

void TestVisitDistribution() {
  using az::Gomoku;
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  FlatEvaluator evaluator;
  az::MctsConfig config;
  config.simulation_num_ = 60;
  config.dirichlet_epsilon_ = 0.0f;
  az::Mcts mcts;
  std::mt19937 rng(2);
  std::vector<int> visit_action, visit_count;
  mcts.Search(game, config, evaluator, rng, visit_action, visit_count);

  std::vector<float> pi(Gomoku::kActionNum, 0.0f);
  az::Mcts::VisitDistribution(visit_action, visit_count, pi.data());
  float sum = 0.0f;
  for (float p : pi) sum += p;
  CHECK(std::fabs(sum - 1.0f) < 1e-4f || sum == 0.0f);
  CHECK(pi[A(7, 7)] == 0.0f && pi[A(8, 8)] == 0.0f); // occupied cells get none
}

void TestMctsTreeReuse() {
  using az::Gomoku;
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 80;
  config.dirichlet_epsilon_ = 0.0f;
  config.reuse_tree_ = true;
  config.max_retained_nodes_ = 1000;
  config.max_retained_edges_ = 10000;
  CountingEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(11);
  std::vector<int> actions, visits;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.last_search_reused());

  int best = -1, best_count = -1;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    if (visits[i] > best_count) {
      best = actions[i];
      best_count = visits[i];
    }
  }
  CHECK(best >= 0 && game.IsLegal(best));
  CHECK(mcts.AdvanceRoot(best));
  CHECK(game.Apply(best));
  CHECK(mcts.node_count() <=
        static_cast<std::size_t>(config.max_retained_nodes_));
  CHECK(mcts.edge_count() <=
        static_cast<std::size_t>(config.max_retained_edges_));
  const int inherited_visits = mcts.root_visits();
  CHECK(inherited_visits > 0);

  evaluator.calls = 0;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(mcts.last_search_reused());
  CHECK(mcts.root_visits() >= inherited_visits + config.simulation_num_);
  const int reused_calls = evaluator.calls;

  CountingEvaluator fresh_evaluator;
  az::Mcts fresh;
  fresh.Search(game, config, fresh_evaluator, rng, actions, visits);
  CHECK(!fresh.last_search_reused());
  CHECK(reused_calls < fresh_evaluator.calls);

  Gomoku unrelated;
  CHECK(PlayMoves(unrelated, {A(0, 0), A(14, 14)}));
  mcts.Search(unrelated, config, evaluator, rng, actions, visits);
  CHECK(!mcts.last_search_reused());
}

void TestMctsReuseDisabledByDefault() {
  az::Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 20;
  config.dirichlet_epsilon_ = 0.0f;
  CHECK(!config.reuse_tree_);
  CountingEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(17);
  std::vector<int> actions, visits;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(mcts.root_visits() == 20);
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.last_search_reused());
  CHECK(mcts.root_visits() == 20); // did not accumulate across calls
}

void TestMctsReuseSafetyAndBounds() {
  using az::Gomoku;
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 40;
  config.dirichlet_epsilon_ = 0.0f;
  config.reuse_tree_ = true;
  config.max_retained_nodes_ = 500;
  config.max_retained_edges_ = 5000;
  CountingEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(29);
  std::vector<int> actions, visits;

  // Public AdvanceRoot must reject malformed external input without indexing
  // the board out of range.
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.AdvanceRoot(-1));
  CHECK(mcts.node_count() == 0);
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.AdvanceRoot(Gomoku::kActionNum));
  CHECK(mcts.node_count() == 0);

  // Exercise many root advances. After every real move the retained tree is
  // either within both hard budgets or was dropped completely for rebuild.
  for (int move = 0; move < 50 && !game.IsTerminal(); ++move) {
    mcts.Search(game, config, evaluator, rng, actions, visits);
    int best = -1, best_count = -1;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (visits[i] > best_count) {
        best = actions[i];
        best_count = visits[i];
      }
    }
    CHECK(best >= 0 && game.IsLegal(best));
    const bool retained = mcts.AdvanceRoot(best);
    CHECK(game.Apply(best));
    if (retained) {
      CHECK(mcts.node_count() <=
            static_cast<std::size_t>(config.max_retained_nodes_));
      CHECK(mcts.edge_count() <=
            static_cast<std::size_t>(config.max_retained_edges_));
    } else {
      CHECK(mcts.node_count() == 0);
      CHECK(mcts.edge_count() == 0);
    }
  }

  az::MctsConfig tight = config;
  tight.simulation_num_ = 80;
  tight.max_retained_nodes_ = 10;
  tight.max_retained_edges_ = 5000;
  az::Mcts bounded;
  Gomoku bounded_game;
  CHECK(PlayMoves(bounded_game, {A(7, 7), A(8, 8)}));
  bounded.Search(bounded_game, tight, evaluator, rng, actions, visits);
  CHECK(bounded.budget_exhausted());
  CHECK(bounded.node_count() <=
        static_cast<std::size_t>(tight.max_retained_nodes_));
  CHECK(bounded.edge_count() <=
        static_cast<std::size_t>(tight.max_retained_edges_));

  az::MctsConfig edge_tight = config;
  edge_tight.simulation_num_ = 80;
  std::vector<int> initial_candidates;
  bounded_game.CandidateActions(initial_candidates);
  edge_tight.max_retained_nodes_ = 1000;
  edge_tight.max_retained_edges_ =
      static_cast<int>(initial_candidates.size()) + 1;
  az::Mcts edge_bounded;
  edge_bounded.Search(bounded_game, edge_tight, evaluator, rng, actions,
                      visits);
  CHECK(edge_bounded.budget_exhausted());
  CHECK(!actions.empty()); // root policy remains usable
  CHECK(edge_bounded.edge_count() <=
        static_cast<std::size_t>(edge_tight.max_retained_edges_));
  int edge_best = actions[0];
  CHECK(!edge_bounded.AdvanceRoot(edge_best)); // exhausted cache is dropped
  CHECK(edge_bounded.node_count() == 0 && edge_bounded.edge_count() == 0);
}

void TestMctsNoiseResetOnReuse() {
  az::Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 0;
  config.reuse_tree_ = true;
  config.normalized_dirichlet_ = true;
  config.dirichlet_epsilon_ = 0.25f;
  FlatEvaluator evaluator;
  az::Mcts mcts;
  std::vector<int> actions;
  std::vector<int> visits;
  std::vector<float> first;
  std::vector<float> second;
  std::mt19937 rng1(31);
  mcts.Search(game, config, evaluator, rng1, actions, visits);
  mcts.RootPriors(actions, first);
  std::mt19937 rng2(31);
  mcts.Search(game, config, evaluator, rng2, actions, visits);
  mcts.RootPriors(actions, second);
  CHECK(mcts.last_search_reused());
  CHECK(first.size() == second.size());
  for (std::size_t i = 0; i < first.size(); ++i)
    CHECK(std::fabs(first[i] - second[i]) < 1e-6f);
}

void TestDirichletNoiseWeight() {
  az::Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 0; // inspect root immediately after expansion
  config.dirichlet_epsilon_ = 0.25f;
  config.dirichlet_alpha_ = 0.3f;
  config.normalized_dirichlet_ = true;
  FlatEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(23), expected_rng(23);
  std::vector<int> actions, visits;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  std::vector<float> prior;
  mcts.RootPriors(actions, prior);
  CHECK(!prior.empty());

  std::gamma_distribution<float> gamma(config.dirichlet_alpha_, 1.0f);
  std::vector<float> noise(prior.size());
  float sum = 0.0f;
  for (float &value : noise) {
    value = gamma(expected_rng);
    sum += value;
  }
  float prior_sum = 0.0f;
  const float base = 1.0f / prior.size();
  for (std::size_t i = 0; i < prior.size(); ++i) {
    const float expected = 0.75f * base + 0.25f * noise[i] / sum;
    CHECK(std::fabs(prior[i] - expected) < 1e-6f);
    prior_sum += prior[i];
  }
  CHECK(std::fabs(prior_sum - 1.0f) < 1e-5f);
}

void TestReplaySaveReportsWriteFailure() {
  az::ReplayBuffer buffer(2);
  az::Sample sample{};
  buffer.Push(sample);
  // Linux's /dev/full accepts open() but reports ENOSPC while flushing. Save
  // must not claim success and replace a valid replay checkpoint afterward.
  CHECK(!buffer.Save("/dev/full"));
}

void TestReplayLoadFailurePreservesBuffer() {
  az::ReplayBuffer buffer(2);
  az::Sample sample{};
  sample.value = 0.5f;
  buffer.Push(sample);
  const char *path = "/tmp/az_bad_replay.bin";
  FILE *out = std::fopen(path, "wb");
  std::size_t bad_size = 99;
  std::size_t bad_position = 0;
  CHECK(out != nullptr);
  if (out != nullptr) {
    std::fwrite(&bad_size, sizeof(bad_size), 1, out);
    std::fwrite(&bad_position, sizeof(bad_position), 1, out);
    std::fclose(out);
  }
  CHECK(!buffer.Load(path));
  CHECK(buffer.Size() == 1); // rejected header never touched live records
  CHECK(std::fabs(buffer.At(0).value - 0.5f) < 1e-6f);
  std::remove(path);

  out = std::fopen(path, "wb");
  std::size_t one = 1;
  std::size_t position = 1;
  CHECK(out != nullptr);
  if (out != nullptr) {
    std::fwrite(&one, sizeof(one), 1, out);
    std::fwrite(&position, sizeof(position), 1, out);
    std::fclose(out); // payload deliberately absent
  }
  CHECK(!buffer.Load(path));
  CHECK(buffer.Size() == 0); // partial payload is never addressable
  std::remove(path);
}

} // namespace

int main() {
  TestHorizontalWin();
  TestVerticalAndDiagonalWin();
  TestIllegalMoves();
  TestDraw();
  TestEncode();
  TestEvalCacheIncludesLastMove();
  TestSymmetryConsistency();
  TestMctsForcedWin();
  TestVisitDistribution();
  TestMctsTreeReuse();
  TestMctsReuseDisabledByDefault();
  TestMctsReuseSafetyAndBounds();
  TestMctsNoiseResetOnReuse();
  TestDirichletNoiseWeight();
  TestReplaySaveReportsWriteFailure();
  TestReplayLoadFailurePreservesBuffer();

  std::printf("%d checks, %d failed\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
