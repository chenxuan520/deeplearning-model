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
  if (reuse_tree_active_ && max_retained_edges_ > 0 &&
      edges_.size() + candidate_scratch_.size() >
          static_cast<std::size_t>(max_retained_edges_)) {
    budget_exhausted_ = true;
    node.edge_begin_ = -1;
    node.edge_num_ = 0;
    return;
  }
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
    edge.base_prior_ = edge.prior_;
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

void Mcts::ApplyRootNoise(int node_index, const MctsConfig &config,
                          std::mt19937 &rng) {
  Node &root = nodes_[node_index];
  for (int i = 0; i < root.edge_num_; ++i) {
    Edge &edge = edges_[root.edge_begin_ + i];
    edge.prior_ = edge.base_prior_;
  }
  if (config.dirichlet_epsilon_ <= 0.0f) return;

  std::gamma_distribution<float> gamma(config.dirichlet_alpha_, 1.0f);
  if (!config.normalized_dirichlet_) {
    // Legacy behavior used by the long-running trainer.  Preserve it unless
    // an evaluation explicitly opts into the standard normalized variant.
    float total = 0.0f;
    for (int i = 0; i < root.edge_num_; ++i) {
      Edge &edge = edges_[root.edge_begin_ + i];
      edge.prior_ = (1.0f - config.dirichlet_epsilon_) * edge.base_prior_ +
                    config.dirichlet_epsilon_ * gamma(rng);
      total += edge.prior_;
    }
    if (total > 0.0f) {
      const float inverse = 1.0f / total;
      for (int i = 0; i < root.edge_num_; ++i)
        edges_[root.edge_begin_ + i].prior_ *= inverse;
    }
    return;
  }

  std::vector<float> noise(root.edge_num_, 0.0f);
  float noise_total = 0.0f;
  for (int i = 0; i < root.edge_num_; ++i) {
    noise[i] = gamma(rng);
    noise_total += noise[i];
  }
  if (noise_total > 0.0f) {
    const float inverse = 1.0f / noise_total;
    for (int i = 0; i < root.edge_num_; ++i) {
      Edge &edge = edges_[root.edge_begin_ + i];
      edge.prior_ =
          (1.0f - config.dirichlet_epsilon_) * edge.base_prior_ +
          config.dirichlet_epsilon_ * noise[i] * inverse;
    }
  }
}

void Mcts::ExpandRoot(int node_index, Gomoku &game, const MctsConfig &config,
                       INetEvaluator &evaluator, std::mt19937 &rng) {
  ExpandNode(node_index, game, evaluator);
  ApplyRootNoise(node_index, config, rng);
}

bool Mcts::SamePosition(const Gomoku &left, const Gomoku &right) {
  return left.current_player() == right.current_player() &&
         left.move_count() == right.move_count() &&
         left.last_action() == right.last_action() &&
         left.Result() == right.Result() && left.board() == right.board();
}

void Mcts::Reset() {
  nodes_.clear();
  edges_.clear();
  path_nodes_.clear();
  path_edges_.clear();
  root_ = -1;
  last_search_reused_ = false;
  reuse_tree_active_ = false;
  budget_exhausted_ = false;
}

void Mcts::ReleaseTreeStorage() {
  std::vector<Node>().swap(nodes_);
  std::vector<Edge>().swap(edges_);
  std::vector<int>().swap(path_nodes_);
  std::vector<int>().swap(path_edges_);
  root_ = -1;
  last_search_reused_ = false;
  reuse_tree_active_ = false;
  budget_exhausted_ = false;
}

bool Mcts::OverRetentionBudget() const {
  return (max_retained_nodes_ > 0 &&
          nodes_.size() >= static_cast<std::size_t>(max_retained_nodes_)) ||
         (max_retained_edges_ > 0 &&
          edges_.size() >= static_cast<std::size_t>(max_retained_edges_));
}

bool Mcts::AdvanceRoot(int action) {
  if (action < 0 || action >= Gomoku::kActionNum || root_ < 0 ||
      !root_game_.IsLegal(action)) {
    Reset();
    return false;
  }
  int child = -1;
  const Node &root = nodes_[root_];
  if (root.edge_begin_ >= 0) {
    for (int i = 0; i < root.edge_num_; ++i) {
      Edge &edge = edges_[root.edge_begin_ + i];
      if (edge.action_ == action) {
        if (edge.child_ < 0) {
          if (max_retained_nodes_ > 0 &&
              nodes_.size() >=
                  static_cast<std::size_t>(max_retained_nodes_)) {
            ReleaseTreeStorage();
            return false;
          }
          edge.child_ = AllocateNode();
        }
        child = edge.child_;
        break;
      }
    }
  }
  if (child < 0 || !root_game_.Apply(action)) {
    Reset();
    return false;
  }
  root_ = child;
  // Treat reuse as a bounded cache. Dropping an oversized tree is cheaper and
  // safer than holding old+compacted copies simultaneously; the next Search
  // rebuilds the current position from the network.
  if (budget_exhausted_ || OverRetentionBudget()) ReleaseTreeStorage();
  return root_ >= 0;
}

int Mcts::root_visits() const {
  return root_ >= 0 ? nodes_[root_].n_ : 0;
}

void Mcts::RootPriors(std::vector<int> &actions,
                      std::vector<float> &priors) const {
  actions.clear();
  priors.clear();
  if (root_ < 0 || nodes_[root_].edge_begin_ < 0) return;
  const Node &root = nodes_[root_];
  for (int i = 0; i < root.edge_num_; ++i) {
    const Edge &edge = edges_[root.edge_begin_ + i];
    actions.push_back(edge.action_);
    priors.push_back(edge.prior_);
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
  const bool can_reuse = config.reuse_tree_ && root_ >= 0 &&
                         SamePosition(root_game_, game);
  last_search_reused_ = can_reuse;
  if (!can_reuse) {
    Reset();
    root_ = AllocateNode();
    root_game_ = game;
  }
  fpu_reduction_ = config.fpu_reduction_;
  max_retained_nodes_ = config.max_retained_nodes_;
  max_retained_edges_ = config.max_retained_edges_;
  reuse_tree_active_ = config.reuse_tree_;
  budget_exhausted_ = false;
  if (config.reuse_tree_) {
    if (max_retained_nodes_ > 0 &&
        nodes_.capacity() < static_cast<std::size_t>(max_retained_nodes_))
      nodes_.reserve(max_retained_nodes_);
    if (max_retained_edges_ > 0 &&
        edges_.capacity() < static_cast<std::size_t>(max_retained_edges_))
      edges_.reserve(max_retained_edges_);
  }
  if (can_reuse && OverRetentionBudget()) {
    ReleaseTreeStorage();
    root_ = AllocateNode();
    root_game_ = game;
    last_search_reused_ = false;
    reuse_tree_active_ = config.reuse_tree_;
    budget_exhausted_ = false;
  }
  Gomoku work = game;
  if (nodes_[root_].edge_begin_ < 0) {
    ExpandRoot(root_, work, config, evaluator, rng);
    // Old unreachable branches may consume the edge budget before a newly
    // selected, unexpanded root can attach its legal moves. Discard the cache
    // and rebuild this exact position instead of returning an empty policy.
    if (budget_exhausted_) {
      ReleaseTreeStorage();
      root_ = AllocateNode();
      root_game_ = game;
      last_search_reused_ = false;
      reuse_tree_active_ = config.reuse_tree_;
      budget_exhausted_ = false;
      work = game;
      ExpandRoot(root_, work, config, evaluator, rng);
    }
  } else {
    ApplyRootNoise(root_, config, rng);
  }

  for (int sim = 0;
       sim < config.simulation_num_ && !budget_exhausted_; ++sim) {
    work = game;
    int node = root_;
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
        if (config.reuse_tree_ && max_retained_nodes_ > 0 &&
            nodes_.size() >=
                static_cast<std::size_t>(max_retained_nodes_)) {
          budget_exhausted_ = true;
          value = 0.0f;
          break;
        }
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
  const Node &root_node = nodes_[root_];
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
