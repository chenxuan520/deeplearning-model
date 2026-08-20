# AlphaZero 五子棋训练档案

> **主线已收口。** 训练在 iter440 完整 gate/checkpoint/replay 边界人工停止，
> 共33,920局去重自对弈（含重启实际35,000局）、约133万局面、87,280次Adam
> 更新。最终 loss 为 policy 2.6492 / value 1.1161；历史 policy 最低为
> 2.6133@iter424。最终图见 `training/policy_loss_analysis.png`，公开内容寻址图为
> `policy_loss_analysis-iter440-288738d2.png`。

> **v1.1.0 low-sim fast 模型**：冻结 iter440 的 trunk/value/BN，仅蒸馏
> policy head；fast iter4 SHA-256
> `1bbd86347ee4942f8b99c9732f8afbd20c896bd5ef703eac87f11f45dedb26ad`。
> 48-sim(reuse) L6黑/白88%/56%、L7 100%/100%；96-sim L6
> 96%/76%、L7 100%/100%；600-sim 对 iter440 直接40:0，L6/L7
> 25局/颜色最大回退4pp。完整实验见 standalone 仓库
> `FAST_POLICY_DISTILLATION.md`。

这份模型的状态、π 与 z 完全由 **AlphaZero/AlphaGo Zero 风格的自对弈**产生：

- 不使用棋谱监督学习；
- 不使用旧站 AI 的落子作为标签；
- 不把启发式估值注入 MCTS；hard curriculum 会用手写三/四连检测器重采样自对弈错题,但不提供动作/价值标签；
- `game-old/gobang-web` 的 1–7 档 AI 只作为最终考官。

## 1. 任务与规则

- 棋盘：15×15；
- 动作：225 个落点；
- 规则：自由五子棋，先形成五连或长连即胜；无禁手；
- 黑棋先行；满盘未五连判和。

状态使用当前行棋方视角编码为 4×15×15：

1. 己方棋子；
2. 对方棋子；
3. 上一手落点；
4. 黑棋行棋时全 1，白棋行棋时全 0。

## 2. 网络结构

```text
4×15×15
  ↓ 3×3 Conv(32) + BN + ReLU
4 × [3×3 Conv(32)+BN+ReLU → 3×3 Conv(32)+BN → residual+ReLU]
  ├─ policy: 1×1 Conv(2)+BN+ReLU → FC(450→225 logits)
  └─ value : 1×1 Conv(1)+BN+ReLU → FC(225→64)+ReLU → FC(64→1)+tanh
```

- 序列化参数量：191,853（不含 BN running mean/variance）；
- 文件：770,380 bytes，Float32；
- 损失：合法手策略交叉熵 + `2 × value MSE`；
- 优化器：AdamW，卷积/全连接权重衰减，BN 与 bias 不衰减。

## 3. AlphaZero 自对弈闭环

```text
当前网络
  → PUCT MCTS（网络给先验 P 和叶子价值 V）
  → 根访问次数归一化为改进策略 π
  → 自对弈保存 (state, π, final result z)
  → Replay Buffer 采样 + 8 对称增广
  → 更新策略价值网络
  → latest 与 best 门禁对打
  → 通过才晋级 best
```

MCTS 公式：

```text
score(s,a) = Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
```

训练阶段根节点加入 Dirichlet 噪声；竞技/网页推理关闭噪声。

## 4. 最终训练配置

```bash
./bin/alphazero train \
  --run-dir runtime \
  --workers 48 \
  --games-per-iter 80 \
  --sims 600 \
  --train-steps 200 \
  --batch 128 \
  --lr 0.001 \
  --wd 0.0001 \
  --value-weight 2 \
  --buffer 200000 \
  --max-moves 200 \
  --temp-moves 6 \
  --seed-hard-prob 0.3 \
  --cpuct 0.8 \
  --dir-eps 0.25 \
  --dir-alpha 0.3 \
  --fpu 0 \
  --gate-every 5 \
  --gate-games 20 \
  --gate-threshold 0.55 \
  --save-buffer-every 10 \
  --trunk 32 \
  --blocks 4 \
  --seed 42
```

同参数重启自动恢复 `latest.net + latest.opt + latest.state + buffer.bin`。

## 5. 长尾强化（仍是纯自对弈）

### 候选点剪枝

空盘只搜天元；其余只搜距离已有棋子 Chebyshev 半径 2 内的空点。
这不是启发式打分，只是动作空间剪枝。原先约 217 个合法点分摊 100 次模拟，
大部分边从未访问，根访问策略 π 近似噪声；剪枝后候选约 25–40 个，训练才进入正轨。

### Hard-negative mining（领域特定 curriculum）

Replay Buffer 中对手已有四连，或败局中对手已有三连的局面进入 hard set。这一步使用手写棋形检测器选择“复习哪些题”，
因此不是对 AlphaGo Zero 仅给规则配方的逐字复刻；但 π/z 仍全部来自自对弈搜索与终局。
每个 batch 约 30% 来自 hard set，重点补防守长尾。

### 残局做种自对弈

约 30% 新局从 replay 中的必防局面继续自对弈。局面、π、z 都来自模型自己的
历史轨迹与最终胜负，不含人类/旧 AI 教师标签。

### Recency sampling

60% 样本按近期指数偏置抽取，其余均匀抽取，使训练跟上当前自对弈分布，
同时保留旧经验避免灾难性遗忘。

## 6. 关键机制修复与调参演进

| 问题 | 现象 | 修复 |
|---|---|---|
| MCTS 终局符号 | 必胜局价值方向反了 | 统一“待行方视角”，回溯每层翻转 |
| FPU 锁死探索 | 首条访问边永久领先 | `fpu_reduction=0` |
| 合法点远多于 sims | π 摊平、杀招漏搜 | 半径 2 候选点 |
| 假随机评估器 | 模型被判 0:20 | 改为真实合法手均匀随机 |
| 防守样本稀缺 | 只攻不守 | hard mining + 残局做种 |
| 温度窗口过长 | 半局都是探索噪声 | `temp_moves 12→6` |
| PUCT 过度广搜 | 防守线访问不够 | `c_puct 1.5→0.8`（训练） |
| BN 状态没复制 | worker 推理漂移 | 参数之外显式复制 running mean/variance |
| cache key 漏上一手 | 相同棋盘不同输入错误命中 | key 加 current player + last_action + board |

搜索模拟数逐步从 100→200→300→500→600。64ch×6block（56 万参数）实验吞吐
仅约 189 eval/s，远低于 32ch×4block 的约 884 eval/s，最终选择小网高频迭代。

## 7. 训练结果

首个稳定验收冠军 `champion_final-348b1b34.net`（600 sims，10 局/颜色）：

| 旧站档位 | 黑棋 | 白棋 |
|---|---:|---:|
| 1 基础 | 100% | 90% |
| 2 防御 | 100% | 50% |
| 3 进攻 | 100% | 80% |
| 4 专家 | 100% | 100% |
| 5 大师 MLP | 100% | 100% |
| 6 老师 αβ | 100% | 90% |
| 7 宗师 MCTS+NN | 100% | 100% |

完整训练继续推进瓶颈，模型卡发布的是已经通过正式验收的稳定冠军，避免最新代际
因对手相性振荡而临时退化。

当前在线试玩模型已切换为 `iter330-c4f41fcb.net`。在相同 48 sims、同噪声、
L1-L6 各 50 局的横评中，iter330 总胜率 64.3%、黑方 85.3%，高于首个正式
冠军的 59.0%/70.7%；首个冠军文件继续保留，便于回归和高预算验收对照。

### 最近 5 次晋升模型横评（48 sims）

最近 5 次成功晋升快照 `iter260/285/300/330/335` 两两循环，每组 50 局、
双方各执黑 25 局，前 6 手按访问分布采样。综合胜率依次为 40.5%、42.0%、
46.0%、60.0%、61.5%；相邻晋升模型均由后者获胜，说明 gate 总体对应真实
能力提升，但 iter335 对 iter330 仅 26:24，20 局 gate 存在明显采样方差。

相同 48 sims、Dirichlet `epsilon=.25/alpha=.3`、相同逐局 seed 下，五个模型
分别对 L1-L6 各打 50 局：iter330 与 iter335 总胜率同为 64.3%，前者黑方
85.3% 更强，后者白方 51.3% 更强。iter300 相比 iter285 有回退，说明训练
代际并非严格单调，保留快照和多对手横评是必要的。完整逐档表见
`training/TRAINING_NOTES.md`。

## 8. 浏览器推理

Cloudflare 仅静态托管参数与 JS：

- `model.json`
- `champion_final-348b1b34.net`
- `alphazero-gomoku-3412a43b.js`

浏览器端直接解析 C++ `.net` 文件，运行 Conv2D、BN、残差网络、策略/价值双头及
PUCT MCTS。没有服务器前向计算。

浏览器 MCTS 使用严格有界的 `SearchSession`：真实落子后继承对应 child subtree，
重开/悔棋通过代数令牌立即取消旧搜索；node/edge 超预算时丢弃缓存并 fresh 重建。
离线压力测试覆盖 8 局 152 手，144 次命中复用，最大 43 nodes / 2785 edges，
强制 GC 后堆增长约 0.25MB。

JS/C++ 一致性测试覆盖 4 个棋盘：

- 最大 policy-logit 误差：`2.67e-5`；
- 最大 value 误差：`1.73e-7`；
- 两组确定性 MCTS 落子完全一致。

运行：

```bash
node tools/test_parity.cjs
```

## 9. 训练记录

- `training/TRAINING_NOTES.md`：完整巡检、故障与调参决策；
- `training/policy_loss_analysis.png`：截至主线收口 iter440 的 policy/value loss、滑动均线、晋级线与拟合；
- `training/policy_loss_history.csv`：逐轮原始数据；
- `training/policy_loss_fit.json`：拟合参数。

## 10. 完整源码复现

权威源码位于独立仓库
[chenxuan520/alphazero-gomoku](https://github.com/chenxuan520/alphazero-gomoku)，
不依赖 PyTorch/TensorFlow；本目录 `source/` 仅保留 v1.0.0 发布时快照：

```bash
git clone https://github.com/chenxuan520/alphazero-gomoku
cd alphazero-gomoku
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./bin/test_az                              # 4411 checks, 0 failures
./bin/alphazero info
```

浏览器前向与 C++ 对齐：

```bash
AZ_PROBE=/path/to/alphazero-gomoku/bin/az_model_probe node tools/test_parity.cjs
```
