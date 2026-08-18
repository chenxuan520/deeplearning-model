#include "mcts/mcts.h"

#include <algorithm>
#include <cmath>

namespace az {

int Mcts::AllocateNode() {
  nodes_.push_back(Node());
  return static_cast<int>(nodes_.size()) - 1;
}

float Mcts::TerminalValue(const Gomoku &game) {
  if (game.Result() == 2) {
    return 0.0f; // draw
  }
  // Value is defined from the perspective of the player TO MOVE. At a
  // win-state terminal that (hypothetical) mover is the loser: on a win,
  // Gomoku::Apply keeps current_player_ == winner, so the loser is the side
  // that did NOT just move. Returning -1 keeps one uniform sign convention:
  // every leaf value is from the would-be mover's perspective and the backup
  // flips the sign once per level it climbs.
  return -1.0f;
}

void Mcts::AttachEdges(int node_index, const Gomoku &game,
                       const float *policy) {
  Node &node = nodes_[node_index];
  node.edge_begin_ = static_cast<int>(edges_.size());
  // Restrict the tree to near-stone candidate moves (radius-2 neighborhood):
  // with 100 sims, spreading over ~217 full-board moves would leave most
  // edges unvisited and turn pi targets into noise. Priors renormalize over
  // the candidate subset.
  game.CandidateActions(candidate_scratch_);
  float prior_sum = 0.0f;
  for (int action : candidate_scratch_) {
    prior_sum += policy[action];
  }
  if (prior_sum <= 0.0f) {
    prior_sum = 1.0f; // uniform fallback
  }
  for (int action : candidate_scratch_) {
    Edge edge;
    edge.action_ = action;
    edge.prior_ = policy[action] / prior_sum;
    edges_.push_back(edge);
  }
  node.edge_num_ = static_cast<int>(edges_.size()) - node.edge_begin_;
}

float Mcts::ExpandNode(int node_index, Gomoku &game,
                       INetEvaluator &evaluator) {
  float value = 0.0f;
  float policy[Gomoku::kActionNum];
  evaluator.Predict(game, policy, value);
  AttachEdges(node_index, game, policy);
  return value;
}

void Mcts::ExpandRoot(Gomoku &game, const MctsConfig &config,
                      INetEvaluator &evaluator, std::mt19937 &rng) {
  ExpandNode(0, game, evaluator);
  if (config.dirichlet_epsilon_ <= 0.0f) {
    return;
  }
  Node &root = nodes_[0];
  std::gamma_distribution<float> gamma(config.dirichlet_alpha_, 1.0f);
  float total = 0.0f;
  for (int i = 0; i < root.edge_num_; ++i) {
    Edge &edge = edges_[root.edge_begin_ + i];
    edge.prior_ = (1.0f - config.dirichlet_epsilon_) * edge.prior_ +
                  config.dirichlet_epsilon_ * gamma(rng);
    total += edge.prior_;
  }
  if (total > 0.0f) {
    const float inverse = 1.0f / total;
    for (int i = 0; i < root.edge_num_; ++i) {
      edges_[root.edge_begin_ + i].prior_ *= inverse;
    }
  }
}

int Mcts::SelectEdge(int node_index, float c_puct) const {
  const Node &node = nodes_[node_index];
  const float parent_q = node.q();
  const float sqrt_n = std::sqrt(static_cast<float>(std::max(1, node.n_)));
  int best = -1;
  float best_score = -1e30f;
  for (int i = 0; i < node.edge_num_; ++i) {
    const Edge &edge = edges_[node.edge_begin_ + i];
    // First-play urgency: unvisited edges get pulled toward the current
    // parent-state estimate minus a reduction, instead of optimism.
    const float q =
        edge.n_ > 0 ? edge.w_ / edge.n_ : parent_q - fpu_reduction_;
    const float u = c_puct * edge.prior_ * sqrt_n / (1.0f + edge.n_);
    const float score = q + u;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return node.edge_begin_ + best;
}

void Mcts::Search(const Gomoku &game, const MctsConfig &config,
                  INetEvaluator &evaluator, std::mt19937 &rng,
                  std::vector<int> &visit_action,
                  std::vector<int> &visit_count) {
  nodes_.clear();
  edges_.clear();
  path_nodes_.clear();
  path_edges_.clear();

  const int root = AllocateNode();
  fpu_reduction_ = config.fpu_reduction_;
  Gomoku work = game;
  ExpandRoot(work, config, evaluator, rng);

  for (int sim = 0; sim < config.simulation_num_; ++sim) {
    work = game;
    int node = root;
    path_nodes_.clear();
    path_edges_.clear();

    // Descend to a leaf.
    float value;
    while (true) {
      if (work.IsTerminal()) {
        value = TerminalValue(work);
        break;
      }
      if (nodes_[node].edge_begin_ < 0) {
        value = ExpandNode(node, work, evaluator);
        break;
      }
      const int edge = SelectEdge(node, config.c_puct_);
      path_nodes_.push_back(node);
      path_edges_.push_back(edge);
      work.Apply(edges_[edge].action_);
      int &child = edges_[edge].child_;
      if (child < 0) {
        child = AllocateNode();
      }
      node = child;
    }

    // Backup: v flips sign every level up the path.
    float v = value;
    for (int i = static_cast<int>(path_nodes_.size()) - 1; i >= 0; --i) {
      v = -v;
      Node &parent = nodes_[path_nodes_[i]];
      Edge &edge = edges_[path_edges_[i]];
      ++edge.n_;
      edge.w_ += v;
      ++parent.n_;
      parent.w_ += v;
    }
  }

  // Export root visit distribution.
  visit_action.clear();
  visit_count.clear();
  const Node &root_node = nodes_[root];
  if (root_node.edge_begin_ >= 0) {
    for (int i = 0; i < root_node.edge_num_; ++i) {
      const Edge &edge = edges_[root_node.edge_begin_ + i];
      visit_action.push_back(edge.action_);
      visit_count.push_back(edge.n_);
    }
  }
}

void Mcts::VisitDistribution(const std::vector<int> &visit_action,
                             const std::vector<int> &visit_count, float *pi) {
  std::fill(pi, pi + Gomoku::kActionNum, 0.0f);
  float total = 0.0f;
  for (int count : visit_count) total += count;
  if (total <= 0.0f) {
    return;
  }
  for (std::size_t i = 0; i < visit_action.size(); ++i) {
    pi[visit_action[i]] = visit_count[i] / total;
  }
}

} // namespace az
