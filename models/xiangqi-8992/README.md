# xiangqi-8992 — 象棋 policy/value ResNet + PUCT

用手写 C++ 深度学习框架训练出的象棋单 checkpoint policy/value 模型。模型本体是
`PolicyValueResNet-192x6`，推理时进入 C++ full-tree PUCT，而不是直接用 policy
argmax 走棋。

当前模型部署在原游戏仓库的 `8992` 端口。这个目录是模型档案入口；完整可执行
复现包、源码快照、训练数据清单和逐位验证脚本位于：

```text
/data00/home/lingchen.judy/temp/chenxuanweb/game/repro/8992/
```

## 一句话结论

- 单一 policy/value checkpoint，无手写残局路由。
- 默认推理为 `128` simulations、`cpuct=1.5` 的 NN-guided PUCT。
- 对 8991 的 50 局 matchup 为 `24-13-13`，总分 61%。
- 对 improved-gentle 六种 noise 共 12 局为 `0-12-0`，未达成“战胜所有难度”。
- Cloudflare Workers 已实测，不适合直接跑 8992 推理。

## 模型规格

| 项 | 值 |
| --- | --- |
| 架构 | `PolicyValueResNet-192x6` |
| 输入 | `[batch, 30, 10, 9]` |
| 动作空间 | 2,086 个规范象棋动作 |
| trunk | 192 channels × 6 residual blocks |
| policy head | 1×1 conv 192→4，flatten 360，FC 360→2086 |
| value head | 1×1 conv 192→2，flatten 180，FC 180→256→1，tanh |
| `parameter_count()` | 4,841,449 |
| AdamW 可训练元素 | 4,838,947 |
| 模型文件 | 19,386,644 bytes |
| optimizer 文件 | 38,712,840 bytes，推理不需要 |
| 模型 SHA-256 | `574708cd51352c5b3087dd3b6ecc601e1f3aed2162217c6c1dc746a5141303e2` |

模型参数主要集中在 12 个 192×192 的 3×3 residual convolution，约占可训练元素
82.3%。policy FC 约 75.3 万参数，value head 很小。最终 value 阶段只训练了
128 个 `value.conv.weight` 新连接。

## 输入特征

30 个平面：

| 平面 | 内容 |
| --- | --- |
| 0–6 | 当前棋盘己方七类棋子 |
| 7–13 | 当前棋盘对方七类棋子 |
| 14–20 | 上一局面己方七类棋子 |
| 21–27 | 上一局面对方七类棋子 |
| 28 | 绝对红方行棋标志，红为 1，黑为 0 |
| 29 | 重复局面计数压缩到 `[0,1]` |

红方保持原坐标，黑方旋转 180° 到当前行棋方视角。动作也按当前行棋方视角编码。
动作表 hash 固定为：

```text
94be604e53701223
```

## 训练设计

8992 的训练不是从随机初始化直接训练一个大模型，而是 function-preserving growth：

1. **融合 8991 双网**
   - 8991 policy 64×4 与 value 64×4 融合为一个 128×4 checkpoint。
   - policy/value 输出误差为 0。

2. **训练 cross connections**
   - 原 policy/value 子网内部权重冻结。
   - 只训练两个 64-channel half 之间的跨连接和 head 跨半区列。
   - 训练数据为人类/社区棋谱，不用 8990/8991 对局标签。

3. **生成 512-PUCT 自博弈数据**
   - 使用 cross step5，16 局，512 simulations。
   - policy target 是访问次数分布，不是 argmax。
   - value target 是自然终局胜负，截断局不进入 value。

4. **无损扩容到 192×6**
   - 源是未 cross 训练的 fused 128×4。
   - 新通道、新 block 初始不影响旧输出。
   - policy/value 前向误差均为 0。

5. **只训练新增 policy 容量**
   - `train-scope=growth-policy`
   - 可训练元素 1,346,304，冻结 3,492,643。
   - 数据混合 512-PUCT policy 与人类/社区 policy。
   - step25 对 8991 为 `19-19-12`，没有过 promotion gate。

6. **只训练新增 value-head 连接**
   - `train-scope=growth-value-head`
   - 只开放 128 个 value 连接，其余 4,838,819 元素冻结。
   - step10 对 8991 为 `24-13-13`，被选为 8992。

关键训练参数见 game 仓库：

```text
repro/8992/LINEAGE.md
repro/8992/TRAINING_DESIGN.md
repro/8992/DATA_AND_SAMPLING.md
```

## 数据

主要数据源：

| 数据 | 作用 |
| --- | --- |
| 10 个 human shard | 基础人类棋谱覆盖 |
| ElephantChess PVP | 授权人类对局，权重更高 |
| human ICCS low/mid | 补中低阶段人类棋谱 |
| 16 局 512-PUCT 自博弈 | policy visit distribution 与自然 value |

最终 growth 阶段 source weights：

```text
32,1,1,1,1,1,1,1,1,1,1,4,3,3
```

其中首个 source 是 PUCT 数据。采样不是按行数直接随机，而是先按 source weight
选择 source，再按 game 均匀选局，再在局内选行。value batch 还按
`decisive-value-rate=0.67` 分成胜负与和棋样本。

完整输入清单位于：

```text
repro/8992/manifests/training_inputs.tsv
```

## 推理

部署使用 C++ `xq_resnet_match_joint`：

```text
model = steps_00010.model.bin
value-model = steps_00010.model.bin
simulations = 128
cpuct = 1.5
network-cache-capacity = 10000
joint_network = true
```

每次 leaf expansion 调用一次 joint network forward，得到 policy logits 和 value。
PUCT 只在合法动作上 softmax，沿路径 backup value 时每层换视角取反。

初始局面实测：

```text
move = 炮二平五
networkEvaluations = 129
nodes = 5045
C++ search ≈ 3.3–3.5s
```

## 部署

### 本机或 VPS

最小推理文件：

```text
steps_00010.model.bin
xq_resnet_match_joint
```

网页部署还需要：

```text
scripts/play_nn_web_server.js
engine/*.js
tmp/nn-play/*
```

实测资源：

| 项 | 数值 |
| --- | --- |
| C++ 加载后 RSS | 约 42.9 MiB |
| 初始搜索峰值 RSS | 约 61.9 MiB |
| 冷加载并退出 | 约 0.097s |
| 128-PUCT 初始局面 | 约 3.3–3.5s，8 线程 |

1C2G 可以跑，但只适合单人低频试玩。多人服务至少需要并发限制，更好的方案是常驻
C++ worker pool。

### Cloudflare Workers

已实测，不适合直接跑 8992 推理。

实际探针：

| 测试 | 结果 |
| --- | --- |
| 最小 Worker | 成功部署 |
| JS CPU 循环 `200,000,000` | HTTP 200，客户端约 2.25s |
| JS CPU 循环 `500,000,000` | HTTP 503，`error code: 1102` |
| 模型 `.bin` 打包进 Worker | 被拒，包大小超限 |
| R2 bucket | 当前账号未启用 R2 |

模型打包结果：

```text
Total Upload: 18932.70 KiB / gzip: 13284.75 KiB
Cloudflare 当前账号限制: 3 MiB
Paid 常见限制: 10 MiB
```

即使启用 R2，也只解决模型存储，不解决 C++→WASM、128-PUCT CPU 时间、无法 fork
子进程、跨请求 cache 等问题。

可行边界：

- Cloudflare Pages 放前端；
- Worker 做鉴权、限流、转发；
- R2 存模型备份；
- native C++ 推理放独立后端。

## 评估

### 对 8991

50 局 paired color-swap：

| 颜色 | 胜 | 负 | 和 | 分数 |
| --- | ---: | ---: | ---: | ---: |
| 总计 | 24 | 13 | 13 | 61% |
| 红方 | 10 | 9 | 6 | 52% |
| 黑方 | 14 | 4 | 7 | 70% |

### 对 improved-gentle

六种 noise，每种两局：

```text
0-12-0
red 0-6
black 0-6
```

所以 8992 是对 8991 matchup 的局部突破，不是通用战胜所有 bot 的模型。

## 可复现性

game 仓库已经验证：

- 从冻结父 checkpoint 重建 final stage，10 个 model/optimizer 逐位一致；
- 从两个 8991 父 checkpoint 重跑完整 8992 分支，70 个产物逐位一致；
- 50 局 arena 与 12 局 gentle 证据可独立重构；
- 冻结源码快照可重编译出 7 个历史 C++ binary。

入口：

```bash
cd /data00/home/lingchen.judy/temp/chenxuanweb/game
repro/8992/reproduce_from_frozen.sh
repro/8992/reproduce_full_branch.sh
```

## 权重与二进制

本 `llm` 仓库不直接收录 8992 权重、optimizer、训练 TSV 或完整复现包二进制。
原因是体积较大，且权威复现包已经在 game 仓库保留。

本目录的 `model/artifacts.tsv` 只记录关键 artifact 的 SHA-256 与语义名称，
用于索引和审计；它不是本地二进制校验清单。

核心权重和部署/复现资产发布在 GitHub Release：

```text
https://github.com/chenxuan520/deeplearning-model/releases/tag/xiangqi-8992-v1
```

Release 包含：

- `steps_00010.model.bin`
- `steps_00010.optimizer.bin`
- `xq_resnet_match_joint`
- `source_snapshot.tar.gz`
- `PACKAGE_SHA256SUMS`
- `model.json`
- `validation.json`

不包含约 7.6GB 训练 TSV。需要完整训练数据迁移时，用 game 仓库的
`repro/8992/scripts/export_portable_bundle.sh` 生成独立便携包。

关键路径：

```text
/data00/home/lingchen.judy/temp/chenxuanweb/game/repro/8992/
/data00/home/lingchen.judy/temp/chenxuanweb/game/data/xq_nn/exp018_human_bootstrap/...
```

如果需要离线迁移，使用：

```bash
repro/8992/scripts/export_portable_bundle.sh /path/to/8992-portable
```

## 许可

- C++/JS 推理与复现脚本：沿用本仓库 MIT License。
- 模型权重：可自由使用。
- 训练数据：人类/社区数据来自授权、自有或公开来源；具体源与 SHA 见 game 仓库
  `repro/8992/manifests/training_inputs.tsv`。
