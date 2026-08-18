#include "game/gomoku.h"

#include <algorithm>

namespace az {

void Gomoku::Reset() {
  board_.fill(0);
  current_player_ = kBlack;
  move_count_ = 0;
  last_action_ = -1;
  result_ = 0;
}

bool Gomoku::Apply(int action) {
  if (result_ != 0 || action < 0 || action >= kCellNum || board_[action] != 0) {
    return false;
  }
  int row, column;
  CellXY(action, row, column);
  board_[action] = static_cast<int8_t>(current_player_);
  ++move_count_;
  last_action_ = action;

  if (CheckWin(row, column)) {
    result_ = current_player_;
  } else if (move_count_ == kCellNum) {
    result_ = 2; // draw
  } else {
    current_player_ = -current_player_;
  }
  return true;
}

bool Gomoku::CheckWin(int row, int column) const {
  static const int kDirections[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  const int8_t color = board_[row * kBoardSize + column];
  for (const auto &dir : kDirections) {
    int count = 1;
    for (int sign = -1; sign <= 1; sign += 2) {
      for (int step = 1; step < 5; ++step) {
        const int r = row + sign * dir[0] * step;
        const int c = column + sign * dir[1] * step;
        if (r < 0 || r >= kBoardSize || c < 0 || c >= kBoardSize ||
            board_[r * kBoardSize + c] != color) {
          break;
        }
        ++count;
      }
    }
    if (count >= 5) {
      return true;
    }
  }
  return false;
}

void Gomoku::Encode(deeplearning::FloatTensor4D &output) const {
  output.Resize(1, kPlaneNum, kBoardSize, kBoardSize, 0.0f);
  EncodeInto(output.data());
}

void Gomoku::EncodeInto(float *output) const {
  float *own = output;                  // plane 0: own stones
  float *opponent = output + kCellNum;  // plane 1: opponent stones
  float *last = output + 2 * kCellNum;  // plane 2: last move marker
  float *side = output + 3 * kCellNum;  // plane 3: side color
  std::fill(output, output + kPlaneNum * kCellNum, 0.0f);
  for (int cell = 0; cell < kCellNum; ++cell) {
    if (board_[cell] == current_player_) {
      own[cell] = 1.0f;
    } else if (board_[cell] == -current_player_) {
      opponent[cell] = 1.0f;
    }
  }
  if (last_action_ >= 0) {
    last[last_action_] = 1.0f;
  }
  if (current_player_ == kBlack) {
    std::fill(side, side + kCellNum, 1.0f);
  }
}

int Gomoku::TransformCell(int row, int column, int symmetry) {
  const int n = kBoardSize;
  const bool mirror = (symmetry & 1) != 0;
  const int rotation = (symmetry >> 1) & 3;
  auto rotate90 = [](int &r, int &c, int size) {
    const int old_r = r; // 90 degrees counter-clockwise
    r = size - 1 - c;
    c = old_r;
  };
  for (int i = 0; i < rotation; ++i) {
    rotate90(row, column, n);
  }
  if (mirror) {
    column = n - 1 - column;
  }
  return row * n + column;
}

int Gomoku::TransformAction(int action, int symmetry) {
  int row, column;
  CellXY(action, row, column);
  return TransformCell(row, column, symmetry);
}

void Gomoku::TransformPlanes(const float *input, float *output, int symmetry) {
  if (symmetry == 0) {
    std::copy(input, input + kPlaneNum * kCellNum, output);
    return;
  }
  for (int plane = 0; plane < kPlaneNum; ++plane) {
    const float *src = input + plane * kCellNum;
    float *dst = output + plane * kCellNum;
    for (int row = 0; row < kBoardSize; ++row) {
      for (int column = 0; column < kBoardSize; ++column) {
        dst[TransformCell(row, column, symmetry)] = src[row * kBoardSize + column];
      }
    }
  }
}

void Gomoku::CandidateActions(std::vector<int> &out, int radius) const {
  out.clear();
  if (move_count_ == 0) {
    out.push_back(kCellNum / 2); // center
    return;
  }
  for (int row = 0; row < kBoardSize; ++row) {
    for (int column = 0; column < kBoardSize; ++column) {
      const int cell = row * kBoardSize + column;
      if (board_[cell] != 0) {
        continue;
      }
      bool near_stone = false;
      for (int dr = -radius; dr <= radius && !near_stone; ++dr) {
        for (int dc = -radius; dc <= radius && !near_stone; ++dc) {
          const int nr = row + dr, nc = column + dc;
          if (nr >= 0 && nr < kBoardSize && nc >= 0 && nc < kBoardSize &&
              board_[nr * kBoardSize + nc] != 0) {
            near_stone = true;
          }
        }
      }
      if (near_stone) {
        out.push_back(cell);
      }
    }
  }
  if (out.empty()) {
    // defensive fallback: every empty cell
    for (int cell = 0; cell < kCellNum; ++cell) {
      if (board_[cell] == 0) out.push_back(cell);
    }
  }
}

void Gomoku::TransformPolicy(const float *input, float *output, int symmetry) {
  if (symmetry == 0) {
    std::copy(input, input + kCellNum, output);
    return;
  }
  for (int row = 0; row < kBoardSize; ++row) {
    for (int column = 0; column < kBoardSize; ++column) {
      output[TransformCell(row, column, symmetry)] =
          input[row * kBoardSize + column];
    }
  }
}

} // namespace az
