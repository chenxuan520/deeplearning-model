#include "arena/js_opponent.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

namespace az {

Subprocess::~Subprocess() { Kill(); }

bool Subprocess::Start(const std::string &command) {
  int in_pipe[2], out_pipe[2];
  if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
    return false;
  }
  pid_t pid = fork();
  if (pid < 0) {
    return false;
  }
  if (pid == 0) {
    // child: stdin <- in_pipe[0], stdout -> out_pipe[1]
    dup2(in_pipe[0], STDIN_FILENO);
    dup2(out_pipe[1], STDOUT_FILENO);
    close(in_pipe[0]);
    close(in_pipe[1]);
    close(out_pipe[0]);
    close(out_pipe[1]);
    execl("/bin/sh", "sh", "-c", command.c_str(), (char *)nullptr);
    _exit(127);
  }
  close(in_pipe[0]);
  close(out_pipe[1]);
  to_child_ = fdopen(in_pipe[1], "w");
  from_child_ = fdopen(out_pipe[0], "r");
  if (to_child_ == nullptr || from_child_ == nullptr) {
    return false;
  }
  pid_ = pid;
  return true;
}

bool Subprocess::Exchange(const std::string &request, std::string &response) {
  if (pid_ <= 0) {
    return false;
  }
  if (std::fwrite(request.data(), 1, request.size(), to_child_) !=
          request.size() ||
      std::fwrite("\n", 1, 1, to_child_) != 1 || std::fflush(to_child_) != 0) {
    return false;
  }
  response.clear();
  char buffer[4096];
  while (true) {
    if (std::fgets(buffer, sizeof(buffer), from_child_) == nullptr) {
      return false;
    }
    response += buffer;
    if (!response.empty() && response.back() == '\n') {
      break;
    }
  }
  return true;
}

void Subprocess::Kill() {
  if (pid_ > 0) {
    int status;
    // reap without blocking forever: SIGTERM then waitpid
    kill(pid_, SIGTERM);
    waitpid(pid_, &status, 0);
    pid_ = -1;
  }
  if (to_child_ != nullptr) {
    std::fclose(to_child_);
    to_child_ = nullptr;
  }
  if (from_child_ != nullptr) {
    std::fclose(from_child_);
    from_child_ = nullptr;
  }
}

// --- JsOpponent ---

bool JsOpponent::Start() {
  // engine is at <repo>/tools/js_engine/engine.js; run from repo root
  return process_.Start("node tools/js_engine/engine.js 2>/dev/null");
}

int JsOpponent::PickMove(const Gomoku &game) {
  // Build the request: their board uses 1=black, 2=white; ours +1/-1.
  std::string request = "{\"board\":[";
  const auto &board = game.board();
  for (int row = 0; row < Gomoku::kBoardSize; ++row) {
    request += '[';
    for (int column = 0; column < Gomoku::kBoardSize; ++column) {
      const int8_t cell = board[row * Gomoku::kBoardSize + column];
      request += cell == Gomoku::kBlack   ? '1'
                 : cell == Gomoku::kWhite ? '2'
                                          : '0';
      if (column + 1 < Gomoku::kBoardSize) request += ',';
    }
    request += ']';
    if (row + 1 < Gomoku::kBoardSize) request += ',';
  }
  request += "],\"me\":";
  request += game.current_player() == Gomoku::kBlack ? '1' : '2';
  request += ",\"level\":" + std::to_string(level_);
  request += '}';

  std::string response;
  if (!process_.Exchange(request, response)) {
    return -1;
  }
  if (response.find("\"error\"") != std::string::npos) {
    return -1;
  }
  // parse {"x":N,"y":N}
  int x = -1, y = -1;
  const std::size_t xp = response.find("\"x\":");
  const std::size_t yp = response.find("\"y\":");
  if (xp == std::string::npos || yp == std::string::npos) {
    return -1;
  }
  x = std::atoi(response.c_str() + xp + 4);
  y = std::atoi(response.c_str() + yp + 4);
  if (x < 0 || x >= Gomoku::kBoardSize || y < 0 || y >= Gomoku::kBoardSize) {
    return -1;
  }
  const int action = x * Gomoku::kBoardSize + y;
  return game.IsLegal(action) ? action : -1;
}

} // namespace az
