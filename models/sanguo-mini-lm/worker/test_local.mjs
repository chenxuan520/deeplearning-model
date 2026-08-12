import { readFileSync } from 'fs';
import { forward, greedyToken, encodePrompt } from '/tmp/opencode/sanguo-worker/public/runner.js';

const manifest = JSON.parse(readFileSync('/tmp/opencode/sanguo-worker/public/manifest.json', 'utf8'));
const buf = readFileSync('/tmp/opencode/sanguo-worker/public/weights.bin');
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const tensors = {};
for (const e of manifest.entries) tensors[e.name] = all.subarray(e.offset, e.offset + e.size);
const charToId = new Map();
manifest.vocabulary.forEach((ch, i) => charToId.set(ch, i));
const model = { cfg: manifest.config, vocab: manifest.vocabulary, charToId, t: tensors };

const cur = encodePrompt(model, '话说天下大势');
let text = '';
for (let i = 0; i < 15; i++) {
  const logits = forward(model, cur);
  const next = greedyToken(logits);
  cur.push(next);
  text += model.vocab[next];
}
console.log('JS greedy:', text);
