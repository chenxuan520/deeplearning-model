(function (global) {
  "use strict";

  const SIZE = 15;
  const CELLS = SIZE * SIZE;
  const PLANES = 4;

  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }

  class BinaryReader {
    constructor(buffer) {
      this.buffer = buffer;
      this.view = new DataView(buffer);
      this.offset = 0;
    }
    bytes(count) {
      const out = new Uint8Array(this.buffer, this.offset, count);
      this.offset += count;
      return out;
    }
    u32() {
      const value = this.view.getUint32(this.offset, true);
      this.offset += 4;
      return value;
    }
    i32() {
      const value = this.view.getInt32(this.offset, true);
      this.offset += 4;
      return value;
    }
    f32() {
      const value = this.view.getFloat32(this.offset, true);
      this.offset += 4;
      return value;
    }
    vector(expected) {
      const low = this.view.getUint32(this.offset, true);
      const high = this.view.getUint32(this.offset + 4, true);
      this.offset += 8;
      const count = high * 4294967296 + low;
      assert(count === expected, `vector size mismatch: expected ${expected}, got ${count}`);
      const copy = new Float32Array(expected);
      copy.set(new Float32Array(this.buffer, this.offset, expected));
      this.offset += expected * 4;
      return copy;
    }
  }

  function packConv(weight, outChannels, inChannels, kernel) {
    const kernelSize = inChannels * kernel * kernel;
    const packed = new Float32Array(kernelSize * outChannels);
    for (let out = 0; out < outChannels; out++) {
      const sourceBase = out * kernelSize;
      for (let k = 0; k < kernelSize; k++) {
        packed[k * outChannels + out] = weight[sourceBase + k];
      }
    }
    return packed;
  }

  function readConv(reader, inChannels, outChannels, kernel, padding) {
    const weight = reader.vector(outChannels * inChannels * kernel * kernel);
    const bias = reader.vector(outChannels);
    return {
      inChannels,
      outChannels,
      kernel,
      padding,
      weight,
      packed: packConv(weight, outChannels, inChannels, kernel),
      bias,
    };
  }

  function readNorm(reader, channels, epsilon) {
    const gamma = reader.vector(channels);
    const beta = reader.vector(channels);
    const mean = reader.vector(channels);
    const variance = reader.vector(channels);
    const scale = new Float32Array(channels);
    const bias = new Float32Array(channels);
    for (let c = 0; c < channels; c++) {
      scale[c] = gamma[c] / Math.sqrt(variance[c] + epsilon);
      bias[c] = beta[c] - mean[c] * scale[c];
    }
    return { gamma, beta, mean, variance, scale, bias };
  }

  function readLinear(reader, inputDim, outputDim) {
    return {
      inputDim,
      outputDim,
      weight: reader.vector(inputDim * outputDim),
      bias: reader.vector(outputDim),
    };
  }

  function parseModel(buffer) {
    const reader = new BinaryReader(buffer);
    const magic = String.fromCharCode(...reader.bytes(8));
    assert(magic === "XQPVRN01", `invalid model magic: ${magic}`);
    const version = reader.u32();
    assert(version === 1, `unsupported model version: ${version}`);
    const config = {
      inputChannels: reader.i32(),
      height: reader.i32(),
      width: reader.i32(),
      trunkChannels: reader.i32(),
      residualBlocks: reader.i32(),
      policyChannels: reader.i32(),
      policySize: reader.i32(),
      valueChannels: reader.i32(),
      valueHidden: reader.i32(),
      threadNum: reader.i32(),
      seed: reader.i32(),
      epsilon: reader.f32(),
      momentum: reader.f32(),
    };
    assert(config.inputChannels === PLANES && config.height === SIZE && config.width === SIZE,
      "model board/input shape mismatch");
    assert(config.policySize === CELLS, "model policy size mismatch");

    const model = { config, blocks: [] };
    model.stemConv = readConv(reader, config.inputChannels, config.trunkChannels, 3, 1);
    model.stemNorm = readNorm(reader, config.trunkChannels, config.epsilon);
    for (let i = 0; i < config.residualBlocks; i++) {
      model.blocks.push({
        conv1: readConv(reader, config.trunkChannels, config.trunkChannels, 3, 1),
        norm1: readNorm(reader, config.trunkChannels, config.epsilon),
        conv2: readConv(reader, config.trunkChannels, config.trunkChannels, 3, 1),
        norm2: readNorm(reader, config.trunkChannels, config.epsilon),
      });
    }
    model.policyConv = readConv(reader, config.trunkChannels, config.policyChannels, 1, 0);
    model.policyNorm = readNorm(reader, config.policyChannels, config.epsilon);
    model.policyLinear = readLinear(reader, config.policyChannels * CELLS, config.policySize);
    model.valueConv = readConv(reader, config.trunkChannels, config.valueChannels, 1, 0);
    model.valueNorm = readNorm(reader, config.valueChannels, config.epsilon);
    model.valueHidden = readLinear(reader, config.valueChannels * CELLS, config.valueHidden);
    model.valueOutput = readLinear(reader, config.valueHidden, 1);
    assert(reader.offset === buffer.byteLength,
      `trailing model bytes: ${buffer.byteLength - reader.offset}`);
    return model;
  }

  // Activations use [position, channel] (NHWC without batch) for contiguous
  // output-channel updates in the convolution inner loop.
  function convAffine(input, height, width, layer, norm, relu) {
    const out = new Float32Array(height * width * layer.outChannels);
    const inChannels = layer.inChannels;
    const outChannels = layer.outChannels;
    const kernel = layer.kernel;
    const padding = layer.padding;
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const outBase = (row * width + col) * outChannels;
        // Every convolution in PolicyValueResNet is initialized with
        // use_bias=false.  The serializer still writes zero-filled bias
        // vectors, but C++ inference ignores them.
        for (let oc = 0; oc < outChannels; oc++) out[outBase + oc] = norm.bias[oc];
        let kernelPosition = 0;
        for (let ic = 0; ic < inChannels; ic++) {
          for (let kr = 0; kr < kernel; kr++) {
            const ir = row + kr - padding;
            for (let kc = 0; kc < kernel; kc++, kernelPosition++) {
              const jc = col + kc - padding;
              if (ir < 0 || ir >= height || jc < 0 || jc >= width) continue;
              const value = input[(ir * width + jc) * inChannels + ic];
              const weightBase = kernelPosition * outChannels;
              for (let oc = 0; oc < outChannels; oc++) {
                out[outBase + oc] += value * layer.packed[weightBase + oc] * norm.scale[oc];
              }
            }
          }
        }
        if (relu) {
          for (let oc = 0; oc < outChannels; oc++) {
            if (out[outBase + oc] < 0) out[outBase + oc] = 0;
          }
        }
      }
    }
    return out;
  }

  function residualForward(input, config, block) {
    const first = convAffine(input, config.height, config.width, block.conv1, block.norm1, true);
    const second = convAffine(first, config.height, config.width, block.conv2, block.norm2, false);
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) out[i] = Math.max(0, input[i] + second[i]);
    return out;
  }

  function flattenNCHW(nhwc, channels) {
    const flat = new Float32Array(CELLS * channels);
    for (let channel = 0; channel < channels; channel++) {
      for (let pos = 0; pos < CELLS; pos++) flat[channel * CELLS + pos] = nhwc[pos * channels + channel];
    }
    return flat;
  }

  function linear(input, layer, relu) {
    const out = new Float32Array(layer.outputDim);
    for (let output = 0; output < layer.outputDim; output++) {
      let sum = layer.bias[output];
      const base = output * layer.inputDim;
      for (let inputIndex = 0; inputIndex < layer.inputDim; inputIndex++) {
        sum += input[inputIndex] * layer.weight[base + inputIndex];
      }
      out[output] = relu && sum < 0 ? 0 : sum;
    }
    return out;
  }

  function encodeBoard(board, currentPlayer, lastAction) {
    const input = new Float32Array(CELLS * PLANES); // temporary NCHW
    for (let cell = 0; cell < CELLS; cell++) {
      if (board[cell] === currentPlayer) input[cell] = 1;
      else if (board[cell] === -currentPlayer) input[CELLS + cell] = 1;
    }
    if (lastAction >= 0) input[2 * CELLS + lastAction] = 1;
    if (currentPlayer === 1) input.fill(1, 3 * CELLS, 4 * CELLS);
    const nhwc = new Float32Array(CELLS * PLANES);
    for (let pos = 0; pos < CELLS; pos++) {
      for (let channel = 0; channel < PLANES; channel++) {
        nhwc[pos * PLANES + channel] = input[channel * CELLS + pos];
      }
    }
    return nhwc;
  }

  function forward(model, board, currentPlayer, lastAction) {
    const c = model.config;
    let trunk = convAffine(encodeBoard(board, currentPlayer, lastAction), SIZE, SIZE,
      model.stemConv, model.stemNorm, true);
    for (const block of model.blocks) trunk = residualForward(trunk, c, block);

    const policyTensor = convAffine(trunk, SIZE, SIZE, model.policyConv, model.policyNorm, true);
    const policyLogits = linear(flattenNCHW(policyTensor, c.policyChannels), model.policyLinear, false);

    const valueTensor = convAffine(trunk, SIZE, SIZE, model.valueConv, model.valueNorm, true);
    const hidden = linear(flattenNCHW(valueTensor, c.valueChannels), model.valueHidden, true);
    const rawValue = linear(hidden, model.valueOutput, false)[0];
    return { policyLogits, value: Math.tanh(rawValue) };
  }

  function legalSoftmax(logits, board) {
    const policy = new Float32Array(CELLS);
    let max = -Infinity;
    for (let a = 0; a < CELLS; a++) if (board[a] === 0 && logits[a] > max) max = logits[a];
    let sum = 0;
    for (let a = 0; a < CELLS; a++) {
      if (board[a] !== 0) continue;
      const value = Math.exp(logits[a] - max);
      policy[a] = value;
      sum += value;
    }
    if (sum > 0) for (let a = 0; a < CELLS; a++) policy[a] /= sum;
    return policy;
  }

  function candidateActions(board, moveCount, radius = 2) {
    if (moveCount === 0) return [Math.floor(CELLS / 2)];
    const result = [];
    for (let row = 0; row < SIZE; row++) {
      for (let col = 0; col < SIZE; col++) {
        const cell = row * SIZE + col;
        if (board[cell] !== 0) continue;
        let near = false;
        for (let dr = -radius; dr <= radius && !near; dr++) {
          for (let dc = -radius; dc <= radius && !near; dc++) {
            const r = row + dr, c = col + dc;
            if (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[r * SIZE + c] !== 0) near = true;
          }
        }
        if (near) result.push(cell);
      }
    }
    if (result.length === 0) for (let a = 0; a < CELLS; a++) if (board[a] === 0) result.push(a);
    return result;
  }

  function checkWin(board, action) {
    const row = Math.floor(action / SIZE), col = action % SIZE, color = board[action];
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (const [dr, dc] of directions) {
      let count = 1;
      for (const sign of [-1, 1]) {
        for (let step = 1; step < 5; step++) {
          const r = row + sign * dr * step, c = col + sign * dc * step;
          if (r < 0 || r >= SIZE || c < 0 || c >= SIZE || board[r * SIZE + c] !== color) break;
          count++;
        }
      }
      if (count >= 5) return true;
    }
    return false;
  }

  function cloneState(state) {
    return {
      board: new Int8Array(state.board),
      currentPlayer: state.currentPlayer,
      moveCount: state.moveCount,
      lastAction: state.lastAction,
      result: state.result,
    };
  }

  function applyMove(state, action) {
    if (state.result !== 0 || action < 0 || action >= CELLS || state.board[action] !== 0) return false;
    state.board[action] = state.currentPlayer;
    state.moveCount++;
    state.lastAction = action;
    if (checkWin(state.board, action)) state.result = state.currentPlayer;
    else if (state.moveCount === CELLS) state.result = 2;
    else state.currentPlayer = -state.currentPlayer;
    return true;
  }

  function newNode() {
    return { edges: null, n: 0, w: 0 };
  }

  function sameState(left, right) {
    if (!left || !right || left.currentPlayer !== right.currentPlayer ||
        left.moveCount !== right.moveCount || left.lastAction !== right.lastAction ||
        left.result !== right.result) return false;
    for (let i = 0; i < CELLS; i++) {
      if (left.board[i] !== right.board[i]) return false;
    }
    return true;
  }

  class SearchSession {
    constructor(options = {}) {
      this.maxNodes = Math.max(1, options.maxNodes || 12000);
      // A legal root can contain every board action. The edge budget must
      // always accommodate one usable root even when callers request less.
      this.maxEdges = Math.max(CELLS, options.maxEdges || 250000);
      this.epoch = 0;
      this.reset();
    }

    reset() {
      this.epoch++;
      this.root = null;
      this.rootState = null;
      this.nodeCount = 0;
      this.edgeCount = 0;
      this.budgetExhausted = false;
    }

    begin(rootState) {
      this.epoch++;
      const reused = this.root != null && sameState(this.rootState, rootState);
      if (!reused) {
        this.root = newNode();
        this.rootState = cloneState(rootState);
        this.nodeCount = 1;
        this.edgeCount = 0;
      }
      // Matches C++ Search(): exhaustion applies to one search only. If the
      // retained tree has no room, this search will set it again and the next
      // real-move advance drops the cache.
      this.budgetExhausted = false;
      return { root: this.root, token: this.epoch, reused };
    }

    cancelled(token, externalStop) {
      return token !== this.epoch || (externalStop && externalStop());
    }

    allocateNode() {
      if (this.nodeCount >= this.maxNodes) {
        this.budgetExhausted = true;
        return null;
      }
      this.nodeCount++;
      return newNode();
    }

    reserveEdges(count) {
      if (this.edgeCount + count > this.maxEdges) {
        this.budgetExhausted = true;
        return false;
      }
      this.edgeCount += count;
      return true;
    }

    recountReachable() {
      if (!this.root) return false;
      let nodes = 0, edges = 0;
      const stack = [this.root];
      const seen = new Set();
      while (stack.length) {
        const node = stack.pop();
        if (seen.has(node)) continue;
        seen.add(node);
        nodes++;
        if (nodes > this.maxNodes) return false;
        if (!node.edges) continue;
        edges += node.edges.length;
        if (edges > this.maxEdges) return false;
        for (const edge of node.edges) if (edge.child) stack.push(edge.child);
      }
      this.nodeCount = nodes;
      this.edgeCount = edges;
      return true;
    }

    advance(action) {
      const exhausted = this.budgetExhausted;
      if (!this.root || !this.rootState || !applyMove(this.rootState, action)) {
        this.reset();
        return false;
      }
      const edge = this.root.edges && this.root.edges.find((item) => item.action === action);
      if (!edge) {
        this.reset();
        return false;
      }
      if (!edge.child) edge.child = this.allocateNode();
      if (!edge.child) {
        this.reset();
        return false;
      }
      this.root = edge.child;
      this.epoch++;
      if (exhausted) {
        this.reset();
        return false;
      }
      if (!this.recountReachable()) {
        this.reset();
        return false;
      }
      return true;
    }

    align(state) {
      if (sameState(this.rootState, state)) return true;
      if (!this.rootState || state.moveCount !== this.rootState.moveCount + 1 ||
          state.lastAction < 0) return false;
      const next = cloneState(this.rootState);
      if (!applyMove(next, state.lastAction) || !sameState(next, state)) return false;
      return this.advance(state.lastAction);
    }
  }

  function evaluate(model, state) {
    const output = forward(model, state.board, state.currentPlayer, state.lastAction);
    return { policy: legalSoftmax(output.policyLogits, state.board), value: output.value };
  }

  function expand(node, model, state, session) {
    const result = evaluate(model, state);
    const actions = candidateActions(state.board, state.moveCount);
    if (session && !session.reserveEdges(actions.length)) return result.value;
    let sum = 0;
    for (const action of actions) sum += result.policy[action];
    node.edges = actions.map((action) => ({
      action,
      prior: sum > 0 ? result.policy[action] / sum : 0,
      n: 0,
      w: 0,
      child: null,
    }));
    return result.value;
  }

  function terminalValue(state) {
    return state.result === 2 ? 0 : -1;
  }

  async function search(model, rootState, options = {}) {
    const simulations = Math.max(1, options.simulations || 60);
    const cPuct = options.cPuct == null ? 1.5 : options.cPuct;
    const fpu = options.fpu || 0;
    const yieldEvery = Math.max(1, options.yieldEvery || 4);
    const session = options.session || null;
    if (session && !sameState(session.rootState, rootState)) session.align(rootState);
    const started = session ? session.begin(rootState) :
      { root: newNode(), token: 0, reused: false };
    const root = started.root;
    const inheritedVisits = root.n;
    if (root.edges == null) expand(root, model, cloneState(rootState), session);

    for (let sim = 0; sim < simulations; sim++) {
      if ((session && session.cancelled(started.token, options.shouldStop)) ||
          (!session && options.shouldStop && options.shouldStop())) {
        return { cancelled: true, action: -1, visits: [], root };
      }
      if (session && session.budgetExhausted) break;
      const state = cloneState(rootState);
      let node = root;
      const path = [];
      let leafValue;
      while (true) {
        if (state.result !== 0) {
          leafValue = terminalValue(state);
          break;
        }
        if (node.edges == null) {
          leafValue = expand(node, model, state, session);
          break;
        }
        const parentQ = node.n > 0 ? node.w / node.n : 0;
        const sqrtN = Math.sqrt(Math.max(1, node.n));
        let best = node.edges[0], bestScore = -Infinity;
        for (const edge of node.edges) {
          const q = edge.n > 0 ? edge.w / edge.n : parentQ - fpu;
          const u = cPuct * edge.prior * sqrtN / (1 + edge.n);
          const score = q + u;
          if (score > bestScore) {
            bestScore = score;
            best = edge;
          }
        }
        path.push([node, best]);
        applyMove(state, best.action);
        if (best.child == null) {
          best.child = session ? session.allocateNode() : newNode();
          if (best.child == null) {
            leafValue = 0;
            break;
          }
        }
        node = best.child;
      }
      let value = leafValue;
      for (let i = path.length - 1; i >= 0; i--) {
        value = -value;
        const [parent, edge] = path[i];
        edge.n++;
        edge.w += value;
        parent.n++;
        parent.w += value;
      }
      if ((sim + 1) % yieldEvery === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        if ((session && session.cancelled(started.token, options.shouldStop)) ||
            (!session && options.shouldStop && options.shouldStop())) {
          return { cancelled: true, action: -1, visits: [], root };
        }
      }
    }

    assert(root.edges && root.edges.length, "search root has no candidate moves");
    let best = root.edges[0];
    for (const edge of root.edges) if (edge.n > best.n) best = edge;
    return {
      action: best.action,
      visits: root.edges.map((edge) => ({ action: edge.action, n: edge.n, q: edge.n ? edge.w / edge.n : 0, p: edge.prior })),
      root,
      reused: started.reused,
      inheritedVisits,
      nodeCount: session ? session.nodeCount : null,
      edgeCount: session ? session.edgeCount : null,
      budgetExhausted: session ? session.budgetExhausted : false,
    };
  }

  async function load(manifestUrl) {
    const manifestResponse = await fetch(manifestUrl);
    assert(manifestResponse.ok, `manifest HTTP ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();
    const weightsResponse = await fetch(manifest.file);
    assert(weightsResponse.ok, `weights HTTP ${weightsResponse.status}`);
    const buffer = await weightsResponse.arrayBuffer();
    if (manifest.sha256) {
      assert(global.crypto && global.crypto.subtle, "WebCrypto SHA-256 support required");
      const digest = await global.crypto.subtle.digest("SHA-256", buffer);
      const actual = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
      assert(actual === manifest.sha256, `weights SHA-256 mismatch: ${actual}`);
    }
    const model = parseModel(buffer);
    model.manifest = manifest;
    return model;
  }

  const api = {
    SIZE,
    CELLS,
    parseModel,
    load,
    forward,
    legalSoftmax,
    candidateActions,
    checkWin,
    applyMove,
    search,
    SearchSession,
    createState(board, currentPlayer, lastAction = -1) {
      const cells = board instanceof Int8Array ? new Int8Array(board) : Int8Array.from(board || new Array(CELLS).fill(0));
      let moveCount = 0;
      for (const value of cells) if (value !== 0) moveCount++;
      return { board: cells, currentPlayer: currentPlayer || (moveCount % 2 === 0 ? 1 : -1), moveCount, lastAction, result: 0 };
    },
  };

  global.AlphaZeroGomoku = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
