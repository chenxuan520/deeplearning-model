# sanguo-mini-lm — 三国演义字符级迷你语言模型

用手写 C++ 深度学习框架（仅 `std::vector`，无 BLAS/SIMD）从零训出来的
**~182 万参数**字符级中文语言模型。语料是公有领域公版书《三國志演義》全文，
模型学会了续写半文半白的三国腔文本。

- 在线网页：https://minilm.011203.xyz
- OpenAI 兼容 API：`POST https://minilm.011203.xyz/v1/chat/completions`

## 生成样例

```text
prompt: 话说天下大势  (贪心)
→ ，不可轻动。
  却说曹操在寨中，

prompt: 孔明曰  (temperature 0.7, top-k 10)
→ 却说魏延引五万兵出城，前后

prompt: 却说曹操  (temperature 0.7, top-k 10)
→ 操大惊，急急奔到来，见曹仁军大叫一
```

诚实说明：这是教学框架 + CPU 上的极限产物，能力是"文风模仿"：
人物名、对白引导（曰）、战争叙事词汇都对，但长句逻辑会断片、会循环。
它不是对话模型，只会顺着前缀往下写。

## 模型规格

| 项 | 值 |
|---|---|
| 架构 | Transformer decoder（[框架自带](../../README.md) MiniTransformerLM） |
| tokenizer | `utf8-char`（每个汉字/标点一个 token） |
| 词表 | 3,981（汉字 + `，` `。` `\n`） |
| 隐藏维度 / 头数 | 128 / 4 |
| FFN 维度 / 层数 | 512（ReLU）/ 4 |
| 上下文窗口 | 48 字符 |
| 参数量 | **1,814,157** |

参数拆解：embedding 509,568 + LM 头 513,549 + 4 层 block 793,088（每层：
QKVO 4×128² + FFN 2×128×512 + 两个 LayerNorm）。

## 训练过程详解

### 1. 语料制备（`corpus/` 三份文件即三步中间产物）

1. **下载**：Project Gutenberg [#23950《三國志演義》](https://www.gutenberg.org/ebooks/23950)
   （罗贯中，公有领域），繁体 UTF-8，1.86MB → `sanguo.raw.txt`
2. **清洗**：仓库自带 `tools/clean_text.py --keep-non-ascii` 去 Gutenberg 页眉页脚 → 62.2 万字
3. **繁转简**：OpenCC（opencc-python-reimplemented，`t2s`）→ `sanguo_simplified.txt`
4. **字符集收敛**：只保留 CJK 汉字 + 全角逗号/句号 + 换行（换行充当段落/句子边界），
   其余标点、装饰符号、ASCII 残留全部丢弃 → `sanguo_clean.txt`
   **571,567 字，词表 3,981**

### 2. 模型初始化

```bash
./bin/mini_lm init --model sanguo.param --corpus-file sanguo_clean.txt \
  --tokenizer utf8-char --model-dim 128 --head-num 4 --feed-forward-dim 512 \
  --block-num 4 --context-size 48 --rand-seed 42
```

### 3. 训练策略（与朴素框架斗智斗勇的过程）

机器：64 核 x86 共享服务器（实际可用 ~24 核），纯 CPU。

- **滑窗改切块**：框架默认逐字滑窗采样（stride=1），57 万字 = 57 万样本/epoch，
  每样本 48 个位置全量 softmax（词表 3981），实测要 7.6 天/epoch。
  为此给框架加了 `--sample-stride`（非重叠切块，GPT 预训练标准做法，
  默认 1 不影响旧行为），样本数 ÷48 → **11,907 样本/epoch**
- **Release 构建**：`-O3` 比 Debug 快一个量级
- **两轮训练**：
  - 第一轮 batch 240 × 3 epochs（2h04m），loss 8.29 → 6.24——但 batch 太大，
    每 epoch 只有 ~50 次梯度更新，loss 明显还没下去
  - 第二轮 batch 48（248 步/epoch）从 checkpoint 续训至 12 epochs（7.5h），
    每 epoch 自动 checkpoint

### 4. Loss 曲线（完整日志见 `model/train.log`、`model/train2.log`）

| epoch | loss | perplexity | 阶段 |
|------:|-----:|-----:|------|
| (初始) | 8.29 | ~4000 | 随机初始化的 ln(3981) |
| 1 | 6.66 | 780 | batch 240 大碎步 |
| 2 | 6.25 | 518 | |
| 3 | 6.24 | 515 | 第一轮结束，换 batch 48 续训 |
| 4 | 5.96 | 388 | |
| 5 | 5.03 | 153 | |
| 6 | 4.60 | 99 | |
| 7 | 4.33 | 76 | |
| 8 | 4.12 | 62 | |
| 9 | 3.95 | 52 | |
| 10 | 3.81 | 45 | |
| 11 | 3.70 | 40 | |
| **12** | **3.62** | **37.2** | 收敛于当前配置 |

loss 仍在下降通道中，若继续训（`--resume-checkpoint --epochs 20`）预计还能压一段。

### 5. 训练命令（复现）

```bash
# 工作目录: 框架仓库的 src/，先 ./build.sh false Release
./bin/mini_lm train --model sanguo.param --corpus-file sanguo_clean.txt \
  --epochs 12 --learning-rate 0.001 --batch-size 48 --thread-num 24 \
  --sample-stride 48 --skip-final-eval --progress-every-sec 300
# 中断续训: 加 --checkpoint sanguo.param.ckpt --resume-checkpoint
#           （--epochs 传目标总轮数；checkpoint 不含 Adam 状态，属已知限制）
```

### 6. 本地推理（流式 / 交互）

```bash
./bin/mini_lm generate --model sanguo.param --prompt "玄德曰" \
  --generate-num 60 --temperature 0.7 --top-k 10
# 不传 --prompt 进入交互模式（输一句生成一句，空行/:q 退出）
```

## 部署（Cloudflare Worker，`worker/`）

- `worker/convert_model.py`：`web_model_export` JSON → Float32 二进制
  （`public/weights.bin` 7.26MB + `public/manifest.json`）
- `worker/src/index.js`：1:1 移植 C++ 前向数学的 JS 推理引擎 +
  OpenAI 兼容接口（`/v1/models`、`/v1/chat/completions`，支持
  `temperature` / `top_p` / `max_tokens` / `stream:true`）
- `worker/public/runner.js`：同一份引擎的浏览器版（网页在你本地浏览器推理，
  免费计划友好）
- 一致性验证：`worker/test_local.mjs`（Node 下与 C++ 贪心输出逐字节一致）

部署方式：

```bash
cd worker && CLOUDFLARE_API_TOKEN=<token> npx wrangler deploy
```

curl 示例：

```bash
curl https://minilm.011203.xyz/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"话说天下大势"}],
       "max_tokens":30,"temperature":0.8,"stream":true}'
```

注：这是纯续写模型，messages 的 content 会按行拼接作为续写前缀，没有对话模板。

## 相关代码改动（在框架仓库，未含于本仓库）

- `src/deeplearning/transformer/character_dataset.*`：+`set_sample_stride()`
- `src/demo/mini_lm/main.cpp`：+`--sample-stride`，generate 流式输出与交互模式
- `src/demo/mini_lm/CMakeLists.txt`：GCC 8 链接 `stdc++fs` 修复
- `docs/mini-lm-demo.md`：对应参数文档
