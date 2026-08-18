#pragma once

#include "game/gomoku.h"
#include "train/evaluator.h"

#include <random>
#include <vector>

namespace az {

struct MctsConfig {
  int simulation_num_ = 100;      // tree size per move
  float c_puct_ = 1.5f;           // exploration constant
  float dirichlet_alpha_ = 0.3f;  // root noise shape
  float dirichlet_epsilon_ = 0.25f; // root noise weight (0 disables)
  float fpu_reduction_ = 0.0f;    // first-play-urgency reduction vs parent q
};

// Single-game MCTS over Gomoku with an injected evaluator. Reused across
// moves of one game via Reset(); not thread-safe.
//
// Sign convention: every value v at a node is the expected outcome from the
// perspective of the player TO MOVE at that node; edge statistics w_/q_ are
// stored from the perspective of the player at the parent node (the one
// choosing the action).
class Mcts {
public:
  struct Edge {
    int action_ = -1;
    int child_ = -1;
    float prior_ = 0.0f;
    int n_ = 0;
    float w_ = 0.0f;
  };
  struct Node {
    int edge_begin_ = -1; // index into edges_ (-1: not expanded yet)
    int edge_num_ = 0;
    int n_ = 0;    // total visits through this node
    float w_ = 0.0f; // value sum, perspective of the player to move here
    float q() const { return n_ > 0 ? w_ / n_ : 0.0f; }
  };

  Mcts() : nodes_(), edges_() {
    nodes_.reserve(4096);
    edges_.reserve(4096 * 32);
  }

  // Runs the given number of simulations from the current game position.
  // Fills visit_action / visit_count with per-action visit counts at root.
  void Search(const Gomoku &game, const MctsConfig &config,
              INetEvaluator &evaluator, std::mt19937 &rng,
              std::vector<int> &visit_action, std::vector<int> &visit_count);

  // Visit-count distribution normalized to a probability vector over all
  // kActionNum actions (zeros elsewhere).
  static void VisitDistribution(const std::vector<int> &visit_action,
                                const std::vector<int> &visit_count,
                                float *pi);

private:
  int AllocateNode();
  // Attaches one edge per legal action of the position to the node.
  void AttachEdges(int node_index, const Gomoku &game, const float *policy);
  // Expands a fresh leaf node with the net. Returns value for the player to
  // move at this node.
  float ExpandNode(int node_index, Gomoku &game, INetEvaluator &evaluator);
  void ExpandRoot(Gomoku &game, const MctsConfig &config,
                  INetEvaluator &evaluator, std::mt19937 &rng);
  static float TerminalValue(const Gomoku &game);
  int SelectEdge(int node_index, float c_puct) const;

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  float fpu_reduction_ = 0.25f; // set at Search() entry from config
  // scratch
  std::vector<int> path_nodes_;
  std::vector<int> path_edges_;
  std::vector<int> candidate_scratch_;
};

} // namespace az
