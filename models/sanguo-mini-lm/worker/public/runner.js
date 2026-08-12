// Browser-side inference for the deeplearning-repo MiniTransformerLM
// (decoder, utf8-char) trained on 三国演义. 1:1 port of the C++ forward path.

let MODEL = null;

export async function loadModel(onProgress) {
  if (MODEL) return MODEL;
  const [manifest, wresp] = await Promise.all([
    fetch('/manifest.json').then((r) => r.json()),
    fetch('/weights.bin'),
  ]);
  const total = +wresp.headers.get('content-length') || 0;
  const reader = wresp.body.getReader();
  const chunks = [];
  let got = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress && total) onProgress(got / total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  const all = new Float32Array(buf.buffer);
  const tensors = {};
  for (const e of manifest.entries) {
    tensors[e.name] = all.subarray(e.offset, e.offset + e.size);
  }
  const charToId = new Map();
  manifest.vocabulary.forEach((ch, i) => charToId.set(ch, i));
  MODEL = { cfg: manifest.config, vocab: manifest.vocabulary, charToId, t: tensors };
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

export function forward(model, ids) {
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
      lnRow(x.subarray(xRow, xRow + D), x.subarray(xRow, xRow + D), n2s, n2b, 1e-6);
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

export function greedyToken(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

export function sampleToken(logits, temperature, topK, topP) {
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

export function encodePrompt(model, text) {
  const ids = [];
  for (const ch of text) {
    const id = model.charToId.get(ch);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}
