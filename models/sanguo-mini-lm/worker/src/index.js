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
<html lang="zh">
<head>
<meta charset="utf-8">
<title>三国迷你 LM</title>
<style>
  body { max-width: 680px; margin: 40px auto; font-family: sans-serif; padding: 0 16px; }
  textarea { width: 100%; box-sizing: border-box; }
  #out { white-space: pre-wrap; border: 1px solid #ccc; padding: 12px; min-height: 140px; margin-top: 12px; }
  .row { margin: 8px 0; }
  button { padding: 6px 18px; }
  #status { color: #666; font-size: 13px; }
</style>
</head>
<body>
<h2>三国演义 · 190 万参数迷你语言模型</h2>
<p>模型在你浏览器里推理（约 7MB 权重，首次加载下载一次）。训练语料：《三國志演義》（Project Gutenberg 公版）。它会学三国腔续写，不会"回答问题"。</p>
<div class="row"><textarea id="prompt" rows="2">却说曹操</textarea></div>
<div class="row">
  字数 <input id="num" type="number" value="50" min="1" max="500" style="width:64px">
  temperature <input id="temp" type="number" value="0.8" step="0.1" min="0.1" max="2" style="width:64px">
  top-k <input id="topk" type="number" value="10" min="0" style="width:56px">
  <button id="go">生成</button>
  <label><input id="greedy" type="checkbox"> 贪心</label>
</div>
<div id="status">模型未加载</div>
<div id="out"></div>
<script type="module">
import { loadModel, forward, greedyToken, sampleToken, encodePrompt } from '/runner.js';
const out = document.getElementById('out');
const status = document.getElementById('status');
let model = null;
(async () => {
  try {
    model = await loadModel((p) => { status.textContent = '权重下载中 ' + (p * 100).toFixed(0) + '%'; });
    status.textContent = '模型就绪（约 0.3-0.8 秒/字）';
  } catch (e) {
    status.textContent = '模型加载失败: ' + e.message;
  }
})();
document.getElementById('go').onclick = async () => {
  if (!model) { status.textContent = '模型还在加载中…'; return; }
  out.textContent = '';
  const greedy = document.getElementById('greedy').checked;
  const temperature = +document.getElementById('temp').value || 0.8;
  const topK = +document.getElementById('topk').value || 0;
  const num = Math.min(Math.max(1, +document.getElementById('num').value || 50), 500);
  const cur = encodePrompt(model, document.getElementById('prompt').value);
  if (cur.length === 0) { out.textContent = '(prompt 里没有词表内的字)'; return; }
  out.textContent = '';
  for (let i = 0; i < num; i++) {
    const logits = forward(model, cur);
    const next = greedy ? greedyToken(logits) : sampleToken(logits, temperature, topK, 1);
    cur.push(next);
    out.textContent += model.vocab[next];
    await new Promise((r) => setTimeout(r, 0));
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

    return new Response('not found', { status: 404 });
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
