# deeplearning-model

基于手写 C++ 深度学习框架 [chenxuan520/deeplearning](https://github.com/chenxuan520/deeplearning)
训练并部署的各种微型模型仓库。每个模型一个目录，语料、权重、训练日志、
复现手册、线上部署代码全部收录，可完整复现。

## 模型索引

| 模型 | 任务 | 参数量 | 语料 | 指标 | 在线演示 |
|------|------|--------|------|------|----------|
| [sanguo-mini-lm](models/sanguo-mini-lm/) | 字符级中文语言模型（utf8-char） | ~182 万 | 《三國志演義》简体清洗版 57 万字（Gutenberg #23950，公有领域） | loss 3.62 / ppl 37.2 | [网页 & OpenAI 兼容 API](https://minilm.011203.xyz) |
| [xiangqi-8992](models/xiangqi-8992/) | 象棋 policy/value ResNet + PUCT | 484 万 | 授权/自有/公开象棋棋谱 + 16 局 512-PUCT 自博弈 | vs 8991: 24-13-13；gentle: 0-12-0 | 本地 8992 端口；Cloudflare Worker 不适合直接推理 |
| [alphazero-gomoku](models/alphazero-gomoku/) | 15×15 五子棋 AlphaZero Policy-Value ResNet + PUCT | 19.2 万 | 纯自对弈（无教师数据） | 全谱验收冠军；黑棋全档 100%，白棋 50%–100% | 浏览器本地前向 + MCTS（模型静态托管于 Cloudflare） |

## 目录约定

每个模型目录下统一结构：

```text
models/<model-name>/
├── README.md   # 该模型的完整档案：配置、参数量拆解、训练过程、指标、生成样例、部署、复现命令
├── corpus/     # 语料（raw → 中间产物 → 训练终版，附来源与许可说明）
├── model/      # 训练日志与指标（模型权重、checkpoint 等二进制不入库，按手册复现或另行获取）
└── worker/     # （可选）线上部署工程，如 Cloudflare Worker；同样不含二进制权重
```

新模型入驻照此结构放，并在本文件表格加一行。

## 许可

- 代码（推理/部署脚本）：与上游框架一致，MIT License（见 [LICENSE](LICENSE)）。
- 模型权重：自由使用。语料均为公有领域或自有数据，各模型 README 单独注明。
