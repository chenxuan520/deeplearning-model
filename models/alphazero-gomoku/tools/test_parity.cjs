#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const AZ = require("../public/alphazero-gomoku-b5cd1abe.js");

const ROOT = path.resolve(__dirname, "..");
const MODEL = path.join(ROOT, "public", "champion_final-348b1b34.net");
const PROBE_CANDIDATES = [
  process.env.AZ_PROBE,
  path.join(ROOT, "source", "bin", "az_model_probe"),
  "/data00/home/lingchen.judy/self/alphazero-gomoku/bin/az_model_probe",
].filter(Boolean);
const PROBE = PROBE_CANDIDATES.find((candidate) => fs.existsSync(candidate));
if (!PROBE) {
  throw new Error("az_model_probe not found; build source/ or set AZ_PROBE");
}

async function main() {
  const bytes = fs.readFileSync(MODEL);
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const model = AZ.parseModel(arrayBuffer);
  const cases = [
    "",
    "112",
    "112,113,97,98,82,83",
    "112,111,97,126,82,141,68,69",
  ];

  let worstLogit = 0;
  let worstValue = 0;
  for (const moves of cases) {
    const state = AZ.createState();
    if (moves) for (const token of moves.split(",")) {
      if (!AZ.applyMove(state, Number(token))) throw new Error(`bad test move ${token}`);
    }
    const js = AZ.forward(model, state.board, state.currentPlayer, state.lastAction);
    const cpp = JSON.parse(execFileSync(PROBE, [MODEL, moves], { encoding: "utf8" }));
    let maxLogit = 0;
    for (let i = 0; i < 225; i++) {
      maxLogit = Math.max(maxLogit, Math.abs(js.policyLogits[i] - cpp.logits[i]));
    }
    const valueError = Math.abs(js.value - cpp.value);
    worstLogit = Math.max(worstLogit, maxLogit);
    worstValue = Math.max(worstValue, valueError);
    console.log(JSON.stringify({ moves, maxLogit, valueError, jsValue: js.value, cppValue: cpp.value }));
  }

  if (worstLogit > 2e-4 || worstValue > 2e-5) {
    throw new Error(`parity failed: logits=${worstLogit} value=${worstValue}`);
  }

  // Search parity on small deterministic trees.  The production browser can
  // use more simulations; these fixtures keep test latency bounded.
  for (const [moves, sims] of [["", 12], ["112,113,97,98", 16]]) {
    const state = AZ.createState();
    if (moves) for (const token of moves.split(",")) AZ.applyMove(state, Number(token));
    const js = await AZ.search(model, state, { simulations: sims, yieldEvery: sims + 1 });
    const cpp = JSON.parse(execFileSync(PROBE, [MODEL, moves, String(sims)], { encoding: "utf8" }));
    console.log(JSON.stringify({ moves, sims, jsAction: js.action, cppAction: cpp.mcts_action }));
    if (js.action !== cpp.mcts_action) throw new Error(`MCTS parity failed for ${moves}`);
  }

  console.log(`PASS worstLogit=${worstLogit} worstValue=${worstValue}`);
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
