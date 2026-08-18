// Minimal test harness: CHECK records failures and keeps going.

#include "game/gomoku.h"
#include "mcts/mcts.h"
#include "train/evaluator.h"

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

  std::printf("%d checks, %d failed\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
