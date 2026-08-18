#include "serve/serve.h"

#include "game/gomoku.h"
#include "mcts/mcts.h"
#include "train/evaluator.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

namespace az {

namespace {

// Extracts the integer value following "key" in a flat JSON string.
// ("me":2 / "x":7 — no escaping; board handled separately)
long JsonInt(const std::string &json, const char *key, long fallback) {
  const std::string needle = std::string("\"") + key + "\"";
  std::size_t pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return fallback;
  return std::strtol(json.c_str() + pos + 1, nullptr, 10);
}

// Parses the "board" field's 225 integers (row-major; 0 empty / 1 black /
// 2 white) into cells. Returns parsed int count.
int JsonBoard(const std::string &json, int8_t *cells) {
  const std::size_t begin = json.find("\"board\"");
  if (begin == std::string::npos) return 0;
  const char *p = json.c_str() + begin;
  int count = 0;
  while (*p && count < Gomoku::kCellNum) {
    if (*p >= '0' && *p <= '9') {
      cells[count++] = static_cast<int8_t>(*p - '0');
      ++p;
    } else {
      ++p;
    }
  }
  return count;
}

void SendAll(int fd, const std::string &s) {
  std::size_t sent = 0;
  while (sent < s.size()) {
    // A browser/client can time out while MCTS is still searching.  Do not let
    // a late response to a closed socket raise SIGPIPE and kill the service.
    const ssize_t n =
        send(fd, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
    if (n <= 0) return;
    sent += n;
  }
}

void SendJson(int fd, int status, const std::string &body) {
  const char *status_text = status == 200 ? "200 OK" : "400 Bad Request";
  char head[512];
  std::snprintf(head, sizeof(head),
                "HTTP/1.1 %s\r\nContent-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: content-type\r\n"
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                status_text, body.size());
  SendAll(fd, head);
  SendAll(fd, body);
}

} // namespace

class MoveServer {
public:
  bool LoadModel(const std::string &path, int threads) {
    if (net_.Load(path) != deeplearning::PolicyValueResNet::SUCCESS) {
      std::fprintf(stderr, "[serve] load failed: %s\n", net_.err_msg().c_str());
      return false;
    }
    auto config = net_.config();
    config.thread_num_ = threads;
    if (!evaluator_.Init(config)) {
      std::fprintf(stderr, "[serve] net init failed\n");
      return false;
    }
    if (!AssignWeights(evaluator_.net(), net_)) {
      std::fprintf(stderr, "[serve] weight copy failed\n");
      return false;
    }
    mcts_config_.dirichlet_epsilon_ = 0.0f;
    return true;
  }

  std::string HandleMove(const std::string &body) {
    Gomoku game;
    if (JsonBoard(body, game.MutableBoard().data()) != Gomoku::kCellNum) {
      return "{\"error\":\"bad board\"}";
    }
    const long me = JsonInt(body, "me", 1);
    const int player = me == 1 ? Gomoku::kBlack : Gomoku::kWhite;
    int black_count = 0, white_count = 0;
    for (int i = 0; i < Gomoku::kCellNum; ++i) {
      if (game.board()[i] == 1) ++black_count;
      if (game.board()[i] == 2) ++white_count;
    }
    // their encoding: 1=black / 2=white -> ours: +1/-1
    std::array<int8_t, Gomoku::kCellNum> mapped;
    for (int i = 0; i < Gomoku::kCellNum; ++i) {
      mapped[i] = game.board()[i] == 1 ? Gomoku::kBlack
                  : game.board()[i] == 2 ? Gomoku::kWhite
                                         : 0;
    }
    game.MutableBoard() = mapped;
    int move_count = black_count + white_count;
    // sanity: mover to move must match counts
    if (move_count % 2 == 0 && player != Gomoku::kBlack) {
      return "{\"error\":\"inconsistent mover\"}";
    }
    game.SetState(player, move_count, 0);

    std::vector<int> visit_action, visit_count;
    std::array<float, Gomoku::kActionNum> pi;
    mcts_.Search(game, mcts_config_, evaluator_, rng_, visit_action,
                 visit_count);
    Mcts::VisitDistribution(visit_action, visit_count, pi.data());
    int best = 0;
    float best_p = -1.0f;
    for (int a = 0; a < Gomoku::kActionNum; ++a) {
      if (pi[a] > best_p && game.IsLegal(a)) {
        best_p = pi[a];
        best = a;
      }
    }
    int row, column;
    Gomoku::CellXY(best, row, column);
    char response[64];
    std::snprintf(response, sizeof(response), "{\"x\":%d,\"y\":%d}", row,
                  column);
    return response;
  }

  MctsConfig &mcts_config() { return mcts_config_; }

private:
  deeplearning::PolicyValueResNet net_;
  Evaluator evaluator_;
  MctsConfig mcts_config_;
  Mcts mcts_;
  std::mt19937 rng_{20260816};
};

int ServeMoves(const std::string &model_path, int port, int sims,
               int worker_threads) {
  MoveServer server;
  if (!server.LoadModel(model_path, worker_threads)) {
    return 1;
  }
  server.mcts_config().simulation_num_ = sims;

  const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::perror("socket");
    return 1;
  }
  int one = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
      listen(listen_fd, 16) < 0) {
    std::perror("bind/listen");
    return 1;
  }
  std::fprintf(stderr,
               "[serve] model=%s sims=%d listening on 0.0.0.0:%d\n",
               model_path.c_str(), sims, port);

  while (true) {
    const int client = accept(listen_fd, nullptr, nullptr);
    if (client < 0) continue;

    // read request: headers until \r\n\r\n then body of content-length
    std::string request;
    char buffer[8192];
    std::size_t body_start = std::string::npos;
    std::size_t content_length = 0;
    while (true) {
      const ssize_t n = recv(client, buffer, sizeof(buffer) - 1, 0);
      if (n <= 0) break;
      buffer[n] = '\0';
      request += buffer;
      if (body_start == std::string::npos) {
        body_start = request.find("\r\n\r\n");
        if (body_start != std::string::npos) {
          body_start += 4;
          std::size_t cl = request.find("Content-Length:");
          if (cl == std::string::npos) {
            cl = request.find("content-length:");
          }
          if (cl != std::string::npos) {
            content_length = std::strtoul(
                request.c_str() + cl + 15, nullptr, 10);
          }
        }
      }
      if (body_start != std::string::npos &&
          request.size() >= body_start + content_length) {
        break;
      }
    }

    if (request.rfind("OPTIONS", 0) == 0) {
      SendJson(client, 200, "{}");
    } else if (request.rfind("GET /health", 0) == 0) {
      SendJson(client, 200, "{\"ok\":true}");
    } else if (request.rfind("POST /api/move", 0) == 0 &&
               body_start != std::string::npos) {
      const std::string body = request.substr(body_start, content_length);
      SendJson(client, 200, server.HandleMove(body));
    } else {
      SendJson(client, 400, "{\"error\":\"unsupported\"}");
    }
    close(client);
  }
  return 0;
}

} // namespace az
