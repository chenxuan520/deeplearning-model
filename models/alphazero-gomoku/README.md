# AlphaZero Gomoku Web Model

本目录存放电子书第 25 章浏览器试玩所需的 AlphaZero 五子棋模型。

## 当前发布状态

- 主线训练已在 **iter440** 人工收口：33,920 局去重自对弈（实际35,000局）、
  约133万局面、87,280 Adam steps；停止时 gate 10:10，policy/value loss
  `2.6492 / 1.1161`。
- 最终训练快照：`final_iter440.net`，SHA-256
  `66aa74b70cd2a73b1c61616df13aaa4a61073d3c5cdf10c1979084212827b2c4`；
  收口时 gate best 为 iter435。
- 网页仍发布 **iter330**：它在48-sim多档位横评中更稳；iter415 vs iter360
  在600 sims下25:25，后续 loss 下降未转化为稳定棋力提升。
- 最终训练图：`https://azgomoku.011203.xyz/policy_loss_analysis-iter440-288738d2.png`。

- 权重文件：`public/champion_final-348b1b34.net`（770KB，小模型；按项目约定明确提交到 `deeplearning-model`，不进入框架仓库 `deeplearning`）
- 文件格式：`XQPVRN01` / version 1 / little-endian Float32
- 网络：4 输入平面，32 通道，4 个残差块，15×15；策略头 2×15×15→225，价值头 1×15×15→64→1
- 参数量：191,853
- 文件大小：770,380 bytes
- SHA-256：`348b1b3448ded9f32d7738cb31cbbe3739c819c54e2b20b29e266dc4780345f4`
- C++ 源码：`source/`（可独立 CMake 构建，含规则、网络、MCTS、自对弈、训练器、门禁、测试）

网页端应直接 `fetch()` 二进制文件，按 `public/model.json` 描述解析各层参数，完成
Conv2D / BatchNorm / ReLU / ResidualBlock / policy-value heads 前向，再在
浏览器本地运行 PUCT MCTS；不依赖服务器前向计算。

Cloudflare 静态模型地址由本目录 `worker/` 独立部署。它只负责分发参数，
不运行网络前向或 MCTS。

- Manifest: `https://azgomoku.011203.xyz/model.json`
- Weights: `https://azgomoku.011203.xyz/champion_final-348b1b34.net`
- Browser engine: `https://azgomoku.011203.xyz/alphazero-gomoku-3412a43b.js`
- Browser MCTS：严格有界子树复用；重开/悔棋取消旧搜索，node/edge 超限自动
  fresh 重建。压力测试8局152手、144次命中复用、堆净增约256KB。

快速构建与验证：

```bash
cmake -S source -B source/build -DCMAKE_BUILD_TYPE=Release
cmake --build source/build -j
source/bin/test_az
AZ_PROBE="$PWD/source/bin/az_model_probe" node tools/test_parity.cjs
node --expose-gc tools/test_reuse_stress.cjs
```
