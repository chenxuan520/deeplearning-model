# AlphaZero Gomoku Web Model

本目录存放电子书第 25 章浏览器试玩所需的 AlphaZero 五子棋模型。

## 当前发布状态

- 主线训练已在 **iter440** 人工收口：33,920 局去重自对弈（实际35,000局）、
  约133万局面、87,280 Adam steps；停止时 gate 10:10，policy/value loss
  `2.6492 / 1.1161`。
- 最终训练快照：`final_iter440.net`，SHA-256
  `66aa74b70cd2a73b1c61616df13aaa4a61073d3c5cdf10c1979084212827b2c4`；
  收口时 gate best 为 iter435。
- 网页现发布 **v1.1.0 fast iter4**：以 iter440 为冻结 trunk/value/BN，
  只用600-sim teacher 蒸馏 policy head。48-sim(reuse) L6 总胜率72%，
  96-sim L6为86%，L7@48/96均100%；600 sims下40:0击败iter440。
- 最终训练图：`https://azgomoku.011203.xyz/policy_loss_analysis-iter440-288738d2.png`。

- 当前网页权重：`public/fast-iter4-1bbd8634.net`（770KB；按项目约定明确提交到 `deeplearning-model`，不进入框架仓库 `deeplearning`）
- v1.0.0 稳定冠军与 iter330 仍原样保留，作为高预算/旧网页回归基线。
- 文件格式：`XQPVRN01` / version 1 / little-endian Float32
- 网络：4 输入平面，32 通道，4 个残差块，15×15；策略头 2×15×15→225，价值头 1×15×15→64→1
- 参数量：191,853
- 文件大小：770,380 bytes
- SHA-256：`1bbd86347ee4942f8b99c9732f8afbd20c896bd5ef703eac87f11f45dedb26ad`
- C++ 源码：[chenxuan520/alphazero-gomoku](https://github.com/chenxuan520/alphazero-gomoku)
  （独立构建，含规则、网络、MCTS、自对弈、训练器、门禁与测试）；本目录的
  `source/` 仅保留 v1.0.0 发布时源码快照。

网页端应直接 `fetch()` 二进制文件，按 `public/channels/stable.json` 描述解析各层参数，完成
Conv2D / BatchNorm / ReLU / ResidualBlock / policy-value heads 前向，再在
浏览器本地运行 PUCT MCTS；不依赖服务器前向计算。

Cloudflare 静态模型地址由本目录 `worker/` 独立部署。它只负责分发参数，
不运行网络前向或 MCTS。

- Stable channel: `https://azgomoku.011203.xyz/channels/stable.json`
- Compatibility alias: `https://azgomoku.011203.xyz/model.json`（由 Worker 映射到 stable）
- Weights: `https://azgomoku.011203.xyz/fast-iter4-1bbd8634.net`
- Browser engine: `https://azgomoku.011203.xyz/alphazero-gomoku-3412a43b.js`
- Browser MCTS：严格有界子树复用；重开/悔棋取消旧搜索，node/edge 超限自动
  fresh 重建。压力测试8局152手、144次命中复用、堆净增约256KB。

v1.1.0 fast 模型完整门禁：

- 48 sims + reuse：L6 黑/白88%/56%，L7 100%/100%；
- 96 sims fresh：L6 96%/76%，L7 100%/100%；
- 600 sims：对 iter440 直接40:0；L6/L7 25局/颜色为100/96/100/100；
- 600 sims：对晋升快照iter360同样40:0（执黑/白均20/20）；
- 48-sim标准噪声对 iter330–435 六个晋升代际：五胜一平零负，合计177:123。

后续晋升模型时，电子书与 game-old 不再改代码；它们永远读取 stable 通道。
只需在本目录运行：

```bash
python3 tools/promote_stable.py \
  --model /path/to/candidate.net \
  --label fast-iterN --release vX.Y.Z \
  --variant "description" --deploy
```

脚本会校验模型头、生成内容寻址文件、原子更新 stable manifest、部署 Worker，
并回读线上 manifest/权重验证 SHA 与大小。

快速构建与验证：

```bash
git clone https://github.com/chenxuan520/alphazero-gomoku
cd alphazero-gomoku
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./bin/test_az

# 回到 deeplearning-model/models/alphazero-gomoku 后做 JS/C++ 对齐
AZ_PROBE=/path/to/alphazero-gomoku/bin/az_model_probe node tools/test_parity.cjs
node --expose-gc tools/test_reuse_stress.cjs
```
