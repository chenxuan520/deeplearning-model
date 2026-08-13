// sanguo-mini-lm — Cloudflare Worker for the deeplearning-repo
// MiniTransformerLM (decoder, utf8-char) trained on 三国演义.
//
// Endpoints:
//   GET  /                       demo page (browser-side inference, free plan OK)
//   GET  /v1/models              OpenAI-compatible model list
//   POST /v1/chat/completions    OpenAI-compatible chat completion
//                                (raw continuation model: message contents are
//                                simply joined as the prompt; supports
//                                temperature/top_p/max_tokens/stream)
//   POST /api/generate           internal simple API ({prompt,num,temperature,topK})
//
// NOTE: server-side generation needs Workers Paid (per-request CPU >10ms).
// The demo page runs inference in the visitor's browser instead.

// ---------- engine (port of src/deeplearning/transformer/*.cpp) ----------

let MODEL = null;

async function loadModel(env) {
  if (MODEL) return MODEL;
  const [manifestObj, weightsObj] = await Promise.all([
    env.ASSETS.fetch('https://assets.local/manifest.json'),
    env.ASSETS.fetch('https://assets.local/weights.bin'),
  ]);
  if (!manifestObj.ok || !weightsObj.ok) {
    throw new Error('weights assets missing (manifest.json / weights.bin)');
  }
  const manifest = await manifestObj.json();
  const buf = await weightsObj.arrayBuffer();
  const all = new Float32Array(buf);
  const tensors = {};
  for (const e of manifest.entries) {
    tensors[e.name] = all.subarray(e.offset, e.offset + e.size);
  }
  const charToId = new Map();
  manifest.vocabulary.forEach((ch, i) => charToId.set(ch, i));
  MODEL = {
    cfg: manifest.config, vocab: manifest.vocabulary, charToId, t: tensors,
  };
  return MODEL;
}

function lnRow(x, out, s, b, eps) {
  const dim = x.length;
  let mean = 0;
  for (let i = 0; i < dim; i++) mean += x[i];
  mean /= dim;
  let variance = 0;
  for (let i = 0; i < dim; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= dim;
  const denom = Math.sqrt(variance + eps);
  for (let i = 0; i < dim; i++) out[i] = ((x[i] - mean) / denom) * s[i] + b[i];
}

function matvec(out, w, x, bias) {
  const dim = x.length;
  const rows = out.length;
  for (let o = 0; o < rows; o++) {
    let acc = bias[o];
    const base = o * dim;
    for (let i = 0; i < dim; i++) acc += w[base + i] * x[i];
    out[o] = acc;
  }
}

function forward(model, ids) {
  const { cfg, t } = model;
  const D = cfg.modelDim, H = cfg.headNum, HD = D / H;
  const CTX = cfg.maxContextSize;
  if (ids.length > CTX) ids = ids.slice(ids.length - CTX);
  const n = ids.length;
  const sqrtD = Math.sqrt(D);

  const x = new Float32Array(n * D);
  for (let pos = 0; pos < n; pos++) {
    const eBase = ids[pos] * D;
    for (let i = 0; i < D; i++) {
      let val = t.embedding[eBase + i] * sqrtD;
      if (cfg.usePositionalEncoding) {
        const angle = pos / Math.pow(10000, (i - (i % 2)) / D);
        val += (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
      }
      x[pos * D + i] = val;
    }
  }

  const q = new Float32Array(n * D);
  const k = new Float32Array(n * D);
  const v = new Float32Array(n * D);
  const att = new Float32Array(n * D);
  const attnOut = new Float32Array(n * D);
  const r1 = new Float32Array(n * D);
  const n1 = new Float32Array(n * D);
  const ff1 = new Float32Array(cfg.feedForwardDim);
  const xo = new Float32Array(D);
  const noBiasD = new Float32Array(D);
  const scores = new Float32Array(n);

  for (let blk = 0; blk < cfg.blockNum; blk++) {
    const Wq = t['b' + blk + '.q'], Wk = t['b' + blk + '.k'],
          Wv = t['b' + blk + '.v'], Wo = t['b' + blk + '.o'];
    for (let p = 0; p < n; p++) {
      const row = x.subarray(p * D, (p + 1) * D);
      matvec(q.subarray(p * D, (p + 1) * D), Wq, row, noBiasD);
      matvec(k.subarray(p * D, (p + 1) * D), Wk, row, noBiasD);
      matvec(v.subarray(p * D, (p + 1) * D), Wv, row, noBiasD);
    }
    att.fill(0);
    const invSqrtHD = 1 / Math.sqrt(HD);
    for (let h = 0; h < H; h++) {
      const hs = h * HD;
      for (let row = 0; row < n; row++) {
        let maxScore = -Infinity;
        for (let col = 0; col <= row; col++) {
          let dot = 0;
          for (let i = 0; i < HD; i++) {
            dot += q[row * D + hs + i] * k[col * D + hs + i];
          }
          scores[col] = dot * invSqrtHD;
          if (scores[col] > maxScore) maxScore = scores[col];
        }
        let sum = 0;
        for (let col = 0; col <= row; col++) {
          scores[col] = Math.exp(scores[col] - maxScore);
          sum += scores[col];
        }
        for (let col = 0; col <= row; col++) {
          const wgt = scores[col] / sum;
          for (let i = 0; i < HD; i++) {
            att[row * D + hs + i] += wgt * v[col * D + hs + i];
          }
        }
      }
    }
    for (let p = 0; p < n; p++) {
      matvec(attnOut.subarray(p * D, (p + 1) * D), Wo,
             att.subarray(p * D, (p + 1) * D), noBiasD);
    }
    for (let i = 0; i < n * D; i++) r1[i] = x[i] + attnOut[i];
    const n1s = t['b' + blk + '.n1s'], n1b = t['b' + blk + '.n1b'],
          n2s = t['b' + blk + '.n2s'], n2b = t['b' + blk + '.n2b'],
          W1 = t['b' + blk + '.w1'], B1 = t['b' + blk + '.b1'],
          W2 = t['b' + blk + '.w2'], B2 = t['b' + blk + '.b2'];
    for (let p = 0; p < n; p++) {
      const n1row = n1.subarray(p * D, (p + 1) * D);
      lnRow(r1.subarray(p * D, (p + 1) * D), n1row, n1s, n1b, 1e-6);
      matvec(ff1, W1, n1row, B1);
      for (let i = 0; i < ff1.length; i++) if (ff1[i] < 0) ff1[i] = 0;
      matvec(xo, W2, ff1, B2);
      const xRow = p * D;
      for (let i = 0; i < D; i++) x[xRow + i] = n1row[i] + xo[i];
      lnRow(x.subarray(xRow, xRow + D), x.subarray(xRow, xRow + D), n2s, n2b,
            1e-6);
    }
  }

  const last = x.subarray((n - 1) * D, n * D);
  const V = cfg.vocabSize;
  const logits = new Float32Array(V);
  for (let o = 0; o < V; o++) {
    let acc = t.outputBias[o];
    const base = o * D;
    for (let i = 0; i < D; i++) acc += t.outputWeight[base + i] * last[i];
    logits[o] = acc;
  }
  return logits;
}

function greedyToken(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

function sampleToken(logits, temperature, topK, topP) {
  const V = logits.length;
  const probs = new Float64Array(V);
  let max = -Infinity;
  for (let i = 0; i < V; i++) {
    probs[i] = logits[i] / temperature;
    if (probs[i] > max) max = probs[i];
  }
  let sum = 0;
  for (let i = 0; i < V; i++) {
    probs[i] = Math.exp(probs[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < V; i++) probs[i] /= sum;

  const order = Array.from({ length: V }, (_, i) => i);
  order.sort((a, b) => probs[b] - probs[a]);
  let keep = V;
  if (topK > 0 && topK < keep) keep = topK;
  let cumulative = 0, pKeep = 0;
  for (let i = 0; i < keep; i++) {
    cumulative += probs[order[i]];
    pKeep++;
    if (cumulative >= topP) break;
  }
  keep = Math.max(1, pKeep);
  let filteredSum = 0;
  for (let i = 0; i < keep; i++) filteredSum += probs[order[i]];
  let r = Math.random() * filteredSum;
  for (let i = 0; i < keep; i++) {
    r -= probs[order[i]];
    if (r <= 0) return order[i];
  }
  return order[0];
}

function encodePrompt(model, text) {
  const ids = [];
  for (const ch of text) {
    const id = model.charToId.get(ch);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}

// ---------- openai-compatible helpers ----------

const MODEL_ID = 'sanguo-mini-lm';

function completionChunk(id, created, delta, finish) {
  return {
    id, object: 'chat.completion.chunk', created, model: MODEL_ID,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

function messagesToPrompt(messages) {
  if (!Array.isArray(messages)) return '';
  return messages.map((m) => String(m?.content ?? '')).join('\n');
}

// ---------- http ----------

const HTML_PAGE = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>三国演义 · 迷你语言模型</title>
<style>
  :root {
    --paper: #f6f1e5;
    --paper-dark: #efe7d3;
    --ink: #2b2622;
    --ink-light: #6b5f52;
    --cinnabar: #a63a2b;
    --cinnabar-deep: #8c2f22;
    --line: #d8cdb4;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--paper-dark);
    color: var(--ink);
    font-family: "Songti SC", "STSong", "Noto Serif CJK SC", "Source Han Serif SC", serif;
    line-height: 1.9;
    padding: 40px 16px;
  }
  .scroll {
    max-width: 720px;
    margin: 0 auto;
    background: var(--paper);
    border: 1px solid var(--line);
    box-shadow: 0 2px 24px rgba(60, 45, 20, 0.14), inset 0 0 60px rgba(180, 150, 90, 0.08);
    padding: 44px 48px 36px;
  }
  header { text-align: center; margin-bottom: 8px; }
  h1 {
    font-size: 30px; letter-spacing: 10px; margin: 0 0 4px;
    font-weight: 600; text-indent: 10px;
  }
  .seal {
    display: inline-block; margin: 10px 0 6px;
    background: var(--cinnabar); color: #f9efe0;
    font-size: 13px; letter-spacing: 0; line-height: 1.15;
    padding: 6px 8px; border-radius: 3px;
    box-shadow: 0 1px 4px rgba(140, 47, 34, 0.4);
    writing-mode: vertical-lr;
  }
  .subtitle { color: var(--ink-light); font-size: 14px; margin: 6px 0 0; }
  .divider {
    border: none; border-top: 1px solid var(--line);
    margin: 22px 0; position: relative;
  }
  .divider::after {
    content: "◆"; position: absolute; left: 50%; top: -0.75em;
    transform: translateX(-50%); background: var(--paper);
    padding: 0 14px; color: var(--line); font-size: 11px;
  }
  .note { font-size: 13.5px; color: var(--ink-light); }
  .note code {
    background: var(--paper-dark); padding: 1px 6px;
    border: 1px solid var(--line); border-radius: 3px; font-size: 12.5px;
  }
  label { font-size: 14px; color: var(--ink-light); }
  textarea {
    width: 100%; padding: 12px 14px; resize: vertical;
    font-family: inherit; font-size: 16px; line-height: 1.8;
    color: var(--ink); background: #fbf8ef;
    border: 1px solid var(--line); border-radius: 2px; outline: none;
  }
  textarea:focus { border-color: var(--cinnabar); box-shadow: 0 0 0 2px rgba(166, 58, 43, 0.12); }
  .controls {
    display: flex; flex-wrap: wrap; gap: 10px 18px;
    align-items: center; margin: 14px 0 4px;
  }
  .controls input[type="number"] {
    width: 62px; padding: 5px 8px; font-family: inherit; font-size: 14px;
    color: var(--ink); background: #fbf8ef;
    border: 1px solid var(--line); border-radius: 2px; outline: none;
  }
  .controls input[type="number"]:focus { border-color: var(--cinnabar); }
  .controls .check { display: inline-flex; align-items: center; gap: 5px; }
  .btns { margin-left: auto; display: flex; gap: 10px; }
  button {
    font-family: inherit; font-size: 15px; letter-spacing: 4px; text-indent: 4px;
    padding: 8px 26px; cursor: pointer; border-radius: 2px;
    border: 1px solid var(--cinnabar-deep);
    background: var(--cinnabar); color: #f9efe0;
    transition: background 0.15s ease;
  }
  button:hover:not(:disabled) { background: var(--cinnabar-deep); }
  button:disabled { opacity: 0.55; cursor: default; }
  #stop {
    background: transparent; color: var(--cinnabar-deep);
    border-color: var(--line); display: none;
  }
  #stop:hover { background: var(--paper-dark); }
  #status { font-size: 13px; color: var(--ink-light); min-height: 22px; }
  #status .bar {
    display: inline-block; vertical-align: middle;
    width: 180px; height: 6px; background: var(--paper-dark);
    border: 1px solid var(--line); border-radius: 3px;
    margin-left: 8px; overflow: hidden;
  }
  #status .bar i { display: block; height: 100%; background: var(--cinnabar); width: 0; transition: width 0.2s; }
  #out {
    margin-top: 10px; padding: 22px 26px; min-height: 160px;
    background: #fbf8ef; border: 1px solid var(--line);
    font-size: 17px; white-space: pre-wrap; word-break: break-all;
  }
  #out:empty::before { content: "（此处将逐字生成）"; color: #c0b49a; }
  #out.generating::after {
    content: "▏"; color: var(--cinnabar);
    animation: blink 0.9s steps(1) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
  footer { margin-top: 26px; font-size: 12.5px; color: var(--ink-light); text-align: center; }
  footer a { color: var(--cinnabar); text-decoration: none; border-bottom: 1px dotted var(--cinnabar); }
  details { margin-top: 14px; font-size: 13.5px; }
  summary { cursor: pointer; color: var(--ink-light); }
  pre {
    background: var(--paper-dark); border: 1px solid var(--line);
    padding: 10px 14px; overflow-x: auto; font-size: 12.5px; line-height: 1.6;
  }
  @media (max-width: 640px) {
    body { padding: 0; }
    .scroll { border: none; padding: 28px 20px; min-height: 100vh; box-shadow: none; }
    h1 { font-size: 24px; letter-spacing: 6px; text-indent: 6px; }
  }
</style>
</head>
<body>
<div class="scroll">
  <header>
    <h1>三国演义</h1>
    <div class="seal">迷你<br>语言<br>模型</div>
    <p class="subtitle">手写 C++ 框架 · 从易经纬 · 一百八十二万参数 · 字符级续写</p>
  </header>
  <hr class="divider">
  <p class="note">
    语料取公版《三國志演義》全文（Project Gutenberg，公有领域），凡五十七万字。
    此模型在你<strong>本机浏览器</strong>中推演，无需联网计算；权重约七兆，首次加载一次。
    模型所学乃三国行文腔调，你开个头，它接着往下诌——莫当真事问它。
  </p>
  <textarea id="prompt" rows="2" placeholder="起个头，譬如：却说曹操">却说曹操</textarea>
  <div class="controls">
    <label>字数 <input id="num" type="number" value="80" min="1" max="500"></label>
    <label>温度 <input id="temp" type="number" value="0.8" step="0.1" min="0.1" max="2"></label>
    <label>top-k <input id="topk" type="number" value="10" min="0"></label>
    <label class="check"><input id="greedy" type="checkbox"> 贪心</label>
    <div class="btns">
      <button id="stop">停</button>
      <button id="go">生成</button>
    </div>
  </div>
  <div id="status">模型未加载</div>
  <div id="out"></div>
  <details>
    <summary>API（OpenAI 兼容，服务端推理）</summary>
    <pre>curl https://minilm.011203.xyz/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"话说天下大势"}],
       "max_tokens":50,"temperature":0.8,"stream":true}'</pre>
  </details>
  <footer>
    <a href="https://github.com/chenxuan520/deeplearning-model">deeplearning-model</a> ·
    框架 <a href="https://github.com/chenxuan520/deeplearning">deeplearning</a>
  </footer>
</div>
<script type="module">
import { loadModel, forward, greedyToken, sampleToken, encodePrompt } from '/runner.js';
const out = document.getElementById('out');
const status = document.getElementById('status');
const goBtn = document.getElementById('go');
const stopBtn = document.getElementById('stop');
let model = null;
let aborted = false;

(async () => {
  try {
    status.innerHTML = '权重下载中 <span class="bar"><i id="bar"></i></span>';
    model = await loadModel((p) => {
      const bar = document.getElementById('bar');
      if (bar) bar.style.width = (p * 100).toFixed(0) + '%';
    });
    status.textContent = '模型就绪（本机推演，约 0.2-0.6 秒/字）';
  } catch (e) {
    status.textContent = '模型加载失败：' + e.message;
  }
})();

stopBtn.onclick = () => { aborted = true; };

goBtn.onclick = async () => {
  if (!model) { status.textContent = '模型尚在加载中…'; return; }
  const greedy = document.getElementById('greedy').checked;
  const temperature = +document.getElementById('temp').value || 0.8;
  const topK = +document.getElementById('topk').value || 0;
  const num = Math.min(Math.max(1, +document.getElementById('num').value || 80), 500);
  const cur = encodePrompt(model, document.getElementById('prompt').value);
  if (cur.length === 0) { out.textContent = ''; status.textContent = '开头里没有词表内的字（模型只识汉字与逗号句号）。'; return; }
  aborted = false;
  goBtn.disabled = true;
  stopBtn.style.display = 'inline-block';
  out.textContent = '';
  out.classList.add('generating');
  try {
    for (let i = 0; i < num; i++) {
      if (aborted) break;
      const logits = forward(model, cur);
      const next = greedy ? greedyToken(logits) : sampleToken(logits, temperature, topK, 1);
      cur.push(next);
      out.textContent += model.vocab[next];
      await new Promise((r) => setTimeout(r, 0));
    }
  } finally {
    out.classList.remove('generating');
    goBtn.disabled = false;
    stopBtn.style.display = 'none';
  }
};
</script>
</body>
</html>`;

const CORS_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET, POST, OPTIONS',
  'access-control-allow-headers': 'content-type, authorization',
};

async function handle(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/' && request.method === 'GET') {
      return new Response(HTML_PAGE, {
        headers: { 'content-type': 'text/html; charset=utf-8' },
      });
    }

    if (url.pathname === '/v1/models' && request.method === 'GET') {
      return Response.json({
        object: 'list',
        data: [{
          id: MODEL_ID, object: 'model',
          created: Math.floor(Date.now() / 1000), owned_by: 'local',
        }],
      });
    }

    if (url.pathname === '/api/generate' || url.pathname === '/v1/chat/completions') {
      if (request.method !== 'POST') {
        return new Response('method not allowed', { status: 405 });
      }
      let body;
      try {
        body = await request.json();
      } catch {
        return new Response('invalid JSON body', { status: 400 });
      }
      const model = await loadModel(env);

      const isChat = url.pathname === '/v1/chat/completions';
      const promptText = isChat ? messagesToPrompt(body.messages)
                                : String(body.prompt ?? '');
      const ids = encodePrompt(model, promptText);
      if (ids.length === 0) {
        return Response.json(
            { error: { message: 'prompt is empty after filtering to model vocabulary' } },
            { status: 400 });
      }
      const num = Math.min(Math.max(1, Number(body.num ?? body.max_tokens) || 50), 500);
      const greedy = body.temperature === undefined || body.temperature === null || body.temperature === 0;
      const temperature = greedy ? 1 : Math.min(Math.max(0.05, Number(body.temperature)), 5);
      const topK = Math.max(0, Number(body.topK) || 0);
      const topP = Math.min(Math.max(0.01, Number(body.top_p ?? 1)), 1);
      const wantStream = isChat && body.stream === true;

      // non-streaming: generate everything, respond at once
      if (!wantStream) {
        const cur = ids.slice();
        let text = '';
        for (let i = 0; i < num; i++) {
          const logits = forward(model, cur);
          const next = greedy ? greedyToken(logits)
                              : sampleToken(logits, temperature, topK, topP);
          cur.push(next);
          text += model.vocab[next];
        }
        if (!isChat) {
          return new Response(text, {
            headers: { 'content-type': 'text/plain; charset=utf-8' },
          });
        }
        return Response.json({
          id: 'chatcmpl-' + Math.random().toString(36).slice(2),
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model: MODEL_ID,
          choices: [{
            index: 0,
            message: { role: 'assistant', content: text },
            finish_reason: 'length',
          }],
          usage: {
            prompt_tokens: ids.length,
            completion_tokens: num,
            total_tokens: ids.length + num,
          },
        });
      }

      // streaming (openai SSE protocol)
      const id = 'chatcmpl-' + Math.random().toString(36).slice(2);
      const created = Math.floor(Date.now() / 1000);
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          const cur = ids.slice();
          let i = 0;
          const send = (obj) => controller.enqueue(
              encoder.encode('data: ' + JSON.stringify(obj) + '\n\n'));
          send(completionChunk(id, created, { role: 'assistant' }, null));
          const step = () => {
            if (i >= num) {
              send(completionChunk(id, created, {}, 'length'));
              controller.enqueue(encoder.encode('data: [DONE]\n\n'));
              controller.close();
              return;
            }
            const logits = forward(model, cur);
            const next = greedy ? greedyToken(logits)
                                : sampleToken(logits, temperature, topK, topP);
            cur.push(next);
            send(completionChunk(id, created, { content: model.vocab[next] }, null));
            i++;
            setTimeout(step, 0);
          };
          step();
        },
      });
      return new Response(stream, {
        headers: {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
        },
      });
    }

    // 未命中路由:交回静态资产(manifest.json / weights.bin / runner.js),
    // 由外层统一补 CORS 头,供其他站点跨域 fetch 权重。
    const assetResp = await env.ASSETS.fetch(request);
    if (url.pathname === '/weights.bin') {
      // 权重基本不变:给一年 immutable 缓存,刷新/重开浏览器都直接命中,
      // 不再回源验证。换权重时需改文件名(如 weights.v2.bin)或加 query 版本号。
      const cached = new Response(assetResp.body, assetResp);
      cached.headers.set(
          'cache-control', 'public, max-age=31536000, immutable');
      return cached;
    }
    return assetResp;
}

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    const resp = await handle(request, env);
    const wrapped = new Response(resp.body, resp);
    for (const [k, v] of Object.entries(CORS_HEADERS)) {
      wrapped.headers.set(k, v);
    }
    return wrapped;
  },
};
