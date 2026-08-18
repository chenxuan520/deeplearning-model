# alphazero-gomoku

15×15 自由规则五子棋的 AlphaZero 式自我对弈强化学习项目，纯 CPU 训练。

**状态：已通关** —— 自对弈策略/价值标签（无 teacher 动作/价值标签）打穿 `game-old` 全部 7 档 AI。
发布冠军在上级目录 `../public/champion_final-348b1b34.net`（32ch/4blocks，19 万参数）；
完整训练档案见 `../TRAINING.md` 与 `../training/TRAINING_NOTES.md`。

复现验收：
```bash
./bin/alphazero gauntlet --model ../public/champion_final-348b1b34.net \
    --levels 1,2,3,4,5,6,7 --games 10 --workers 24 --sims 800 \
    --dir-eps 0

# 只测模型执白/黑(例如 L7 后手 20 局)
./bin/alphazero gauntlet --model ../public/champion_final-348b1b34.net \
    --levels 7 --games 20 --workers 8 --sims 96 --color white
```

训练栈完全建立在 deeplearning 仓库的 WIP 组件副本之上（`lib/` 下：
`FloatTensor4D` / `BatchedConv2D` / `BatchNorm2D` / `ResidualBlock2D` /
`FloatLinear` / `PolicyValueResNet` / `PolicyValueLoss` / `FloatAdamW` /
`ThreadPool`），不依赖任何外部 ML 库。

## 训练原理（AlphaZero 主循环）

```
自我对弈(MCTS+当前网络) → 样本 (s, π, z) 进回放池 → 训练网络 → 与历史最优打擂台(胜率≥55%晋级) → 循环
```

- **网络**：策略-价值双流 ResNet。默认 4 卷积 stem + 4 残差块（trunk 32 通道），
  策略头输出 225 格 logits，价值头 tanh 输出 [-1,1]（行棋方视角）。约 19 万参数。
- **棋盘编码**：4 个 15×15 平面 = 己方子 / 对方子 / 上一手位置 / 行棋方颜色。
- **MCTS**：PUCT 选边 + 根节点 Dirichlet 噪声；先手手数内按访问数分布温度采样，之后贪心。
- **训练目标**：`PolicyValueLoss` = 合法手 masked 策略交叉熵 + 价值 MSE；AdamW
  （权重衰减只作用于 conv/linear 权重，BN/bias 不衰减）。
- **数据增广**：棋盘 8 对称（4 旋转 × 镜像），采样时随机取一种同时变换平面与策略目标。
- **评估缓存**：自对弈同轮内共享局面哈希缓存（己方/对方子布局 → policy/value），
  键包含当前行棋方、上一手位置和完整棋盘；20+ 并行对局下开局重复局面直接
  命中，以内存换 CPU。权重更新即失效。

每轮结构（`train` 命令的默认节奏）：

| 阶段 | 默认量 |
| --- | --- |
| 自对弈 | 40 局 × 20 worker，每步 100 次 MCTS 模拟，上限 200 手判和 |
| 训练 | 80 步 × batch 128（采样 + 8 对称增广），AdamW lr=1e-3 |
| 评估 | 每 5 轮：vs 随机棋手 20 局 + 与 best.net 擂台 20 局 |
| 保存 | 每轮 latest.net/opt/state，晋级时 best.net，每 10 轮 buffer.bin |

吞吐参考（24 核保障值）：并发前向约 880 evals/s（40 worker 实测），
单轮自对弈约 6~9 分钟 + 训练约 2 分钟。

## 构建与测试

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
cd .. && ./bin/test_az        # 4100 checks, 0 failed
```

## 命令

```bash
# 正式训练（后台长跑建议 nohup）
./bin/alphazero train --workers 20 --games-per-iter 40 --sims 100 \
    --train-steps 80 --batch 128 --run-dir runtime --seed 42

# 断点续训：同参数重跑即可（自动读 runtime/latest.net + latest.opt + latest.state）
# --no-resume 冷启动；--no-cache 关评估缓存

# 模型 vs 随机棋手（交替执色）
./bin/alphazero eval --model runtime/best.net --games 20 --sims 100 --workers 8

# 两个模型打擂台
./bin/alphazero arena --model-a runtime/latest.net --model-b runtime/best.net \
    --games 50 --sims 48 --temp-moves 6

# 相同逐局根噪声的模型 vs JS 档位横评
./bin/alphazero gauntlet --model runtime/best.net --levels 1,2,3,4,5,6 \
    --games 25 --sims 48 --dir-eps .25 --dir-alpha .3 --seed 4242

# 人机对战（你是白棋 O，输入 "行 列"）
./bin/alphazero play --model runtime/best.net --sims 100

# 网络前向性能 / 规模
./bin/alphazero bench --concurrent 40 --iters 30
./bin/alphazero info
```

## 监控训练

所有结构化日志追加到 `runtime/train.log`（JSON 行，同时打到 stderr）：

```bash
tail -f runtime/train.log
grep eval_random runtime/train.log   # 看 vs 随机胜率变化（弱网默认 0%，起来后应攀升）
grep '"phase":"gate"' runtime/train.log  # 看擂台晋级记录
```

关键字段：
- `avg_moves`：自对弈平均局长。随机初期顶着 200 手上限，变强后会显著缩短（几十手出胜负）——这是最直观的早期变强信号。
- `policy_loss`：从 ln(225)≈5.42 一路降。
- `cache_hit_rate`：评估缓存命中率，开局重复局面多时 30%+ 正常。
- `gate`：`challenger_wins/best_wins/draws/rate`，`promoted` 表示新网络晋级为 best。

可选 loss 曲线分析需要项目私有虚拟环境，并依赖训练后生成的 `runtime/train.log`：

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/pip install -r tools/requirements-analysis.txt
.venv-analysis/bin/python tools/analyze_loss_scipy.py
```

本模型包已提供生成后的 `../training/policy_loss_history.csv`、拟合 JSON 与 PNG；
无需重跑分析即可核对书中曲线。

## 文件布局

```
lib/            框架组件（自 deeplearning 仓 WIP 拷贝，勿删勿改同步方向）
src/game/       Gomoku 规则/编码/8对称
src/mcts/       MCTS (PUCT + Dirichlet 根噪声)
src/train/      评估器(INetEvaluator/缓存) / 回放池 / 自对弈 / 擂台 / 训练器
test/           单元测试(规则/编码/对称/MCTS 必杀局面)
runtime/        训练产物：latest.net best.net latest.opt latest.state buffer.bin train.log
runtime.out     nohup stdout
```

## 设计取舍备忘

- **FPU reduction 默认 0**：>0 时在值几乎平坦的早期局面会锁死探索（首个被访问的边永远领先）。
- **终局价值约定**：赢的局面返回 -1（按"该走棋的一方视角"的约定，终局时没有下一手，行棋方即输家视角），保证 MCTS 回溯逐层翻转的符号一致。
- **worker 网络副本**：组件内部线程池不可多实例并发复用，所以每个自对弈 worker 持有独立网络副本（拷贝含 BN running statistics，不只可训练参数）。
- **无 resign**：v1 未做认输加速，弱网阶段价值信号不可靠，先靠 max-moves 截断。
