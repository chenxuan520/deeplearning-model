#pragma once

#include "cnn/float_tensor.h"

#include <array>
#include <cstdint>
#include <vector>

namespace az {

// 15x15 free-style Gomoku (no forbidden moves). Black moves first, first
// five-in-a-row (or longer) wins, full board without five is a draw.
//
// Conventions used across the whole project:
//   - Player is +1 (black) or -1 (white).
//   - Every state encoding is from the perspective of the player to move,
//     and every recorded value target z is from the perspective of the
//     player who was to move when the record was produced (AlphaZero style).
class Gomoku {
public:
  static constexpr int kBoardSize = 15;
  static constexpr int kCellNum = kBoardSize * kBoardSize;
  static constexpr int kActionNum = kCellNum; // one action per cell
  static constexpr int kPlaneNum = 4; // own, opponent, last move, side color
  static constexpr int kSymmetryNum = 8; // 4 rotations x optional mirror
  static constexpr int kBlack = 1;
  static constexpr int kWhite = -1;

  Gomoku() { Reset(); }

  void Reset();

  // Applies action (cell index 0..224). Returns false for illegal actions.
  bool Apply(int action);

  bool IsLegal(int action) const { return board_[action] == 0; }

  // Direct state injection (used to resume mid-game positions from replay
  // samples when curriculum-seeding self-play). Not for normal play.
  std::array<int8_t, kCellNum> &MutableBoard() { return board_; }
  void SetState(int current_player, int move_count, int result,
                int last_action = -1) {
    current_player_ = current_player;
    move_count_ = move_count;
    result_ = result;
    last_action_ = last_action;
  }

  // Fills `out` with the SEARCH candidate set: empty cells that have at
  // least one stone within Chebyshev distance kRadius (2). Full-board legal
  // moves stay 0..224; this restriction only prunes the MCTS action space
  // (standard gomoku move generation; no heuristic scoring involved).
  // Empty board -> just the center cell.
  void CandidateActions(std::vector<int> &out, int radius = 2) const;

  // 0 = ongoing, +1 = black wins, -1 = white wins, 2 = draw.
  int Result() const { return result_; }
  bool IsTerminal() const { return result_ != 0; }

  int current_player() const { return current_player_; }
  int move_count() const { return move_count_; }
  int last_action() const { return last_action_; }
  const std::array<int8_t, kCellNum> &board() const { return board_; }

  static void CellXY(int action, int &row, int &column) {
    row = action / kBoardSize;
    column = action % kBoardSize;
  }

  // Fills output as [kPlaneNum x 15 x 15] FloatTensor4D (batch=1) from the
  // perspective of the player to move.
  void Encode(deeplearning::FloatTensor4D &output) const;
  // Same encoding into a raw float buffer of kPlaneNum * kCellNum floats.
  void EncodeInto(float *output) const;

  // Symmetry transform s in [0,8): rotation by (s>>1)*90 degrees counter-
  // clockwise, then a horizontal mirror when (s&1). Used consistently for
  // board planes, policy vectors and action indices.
  static int TransformCell(int row, int column, int symmetry);
  static int TransformAction(int action, int symmetry);

  // Transforms planes/policies of exactly one game state (not batched).
  static void TransformPlanes(const float *input, float *output,
                              int symmetry);
  static void TransformPolicy(const float *input, float *output,
                              int symmetry);

private:
  // Returns true if the stone just placed at (row, column) completed a five.
  bool CheckWin(int row, int column) const;

  std::array<int8_t, kCellNum> board_{};
  int current_player_ = kBlack;
  int move_count_ = 0;
  int last_action_ = -1;
  int result_ = 0;
};

} // namespace az
