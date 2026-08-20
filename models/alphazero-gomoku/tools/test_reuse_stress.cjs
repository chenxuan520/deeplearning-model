#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const ENGINE = process.env.AZ_ENGINE ||
  path.join(ROOT, "public", "alphazero-gomoku-3412a43b.js");
const MODEL = process.env.AZ_MODEL ||
  path.join(ROOT, "public", "champion_final-348b1b34.net");
const AZ = require(ENGINE);

function loadModel() {
  const bytes = fs.readFileSync(MODEL);
  return AZ.parseModel(bytes.buffer.slice(
    bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

async function main() {
  const model = loadModel();
  if (global.gc) global.gc();
  const heapStart = process.memoryUsage().heapUsed;
  let searches = 0;
  let moves = 0;
  let maxNodes = 0;
  let maxEdges = 0;
  let reuseHits = 0;

  for (let gameIndex = 0; gameIndex < 8; gameIndex++) {
    const state = AZ.createState();
    const session = new AZ.SearchSession({ maxNodes: 400, maxEdges: 6000 });
    while (state.result === 0 && state.moveCount < 120) {
      const result = await AZ.search(model, state, {
        simulations: 12,
        yieldEvery: 13,
        session,
      });
      if (result.cancelled || result.action < 0)
        throw new Error("unexpected cancelled/empty search");
      if (result.nodeCount > 400 || result.edgeCount > 6000)
        throw new Error(`retention budget escaped: ${result.nodeCount}/${result.edgeCount}`);
      maxNodes = Math.max(maxNodes, result.nodeCount);
      maxEdges = Math.max(maxEdges, result.edgeCount);
      if (result.reused) reuseHits++;
      searches++;
      if (!AZ.applyMove(state, result.action)) throw new Error("illegal MCTS move");
      const advanced = session.advance(result.action);
      if (!advanced && !result.budgetExhausted)
        throw new Error("subtree advance failed before budget exhaustion");
      moves++;
    }
    session.reset();
    if (session.root !== null || session.nodeCount !== 0 || session.edgeCount !== 0)
      throw new Error("reset retained search state");
  }

  if (global.gc) global.gc();
  const heapEnd = process.memoryUsage().heapUsed;
  const growth = heapEnd - heapStart;
  if (growth > 64 * 1024 * 1024)
    throw new Error(`heap grew too much: ${growth}`);
  if (reuseHits === 0) throw new Error("stress run never reused a subtree");
  console.log(JSON.stringify({
    games: 8,
    searches,
    moves,
    reuseHits,
    maxNodes,
    maxEdges,
    heapGrowthBytes: growth,
  }));
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
