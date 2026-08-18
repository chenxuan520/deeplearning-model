// JS opponent engine harness for alphazero-gomoku acceptance testing.
// Runs the ORIGINAL game-old/gobang-web AI code verbatim (levels 1-7) under
// Node. Protocol: JSON lines on stdin/stdout.
//
//   request : {"board": [[0/1/2] x 15 x 15], "me": 1|2, "level": 1..7}
//   response: {"x": <row>, "y": <col>}   or   {"error": "..."}
//
// Levels: 的大概对应 README 的 7 档:
//   1 basic    2 defense    3 attack (heuristic profiles)
//   4 expert (depth-2 minimax)
//   5 master (26K MLP + heuristic mix)
//   6 teacher (depth-3 alpha-beta, EvaluateBoardStrong)
//   7 grandmaster (MCTS 200 sims, NN prior + strong leaf value)

'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const FRONTEND =
    '/home/lingchen.judy/self/game-old/gobang-web/frontend';

// ---- globals the extracted script.js section depends on ------------------
// (indirect eval runs in global scope, so these must live on globalThis)
globalThis.BOARD_SIZE = 15;
globalThis.gameState = {board: null};

globalThis.checkWin = function (x, y, player, board) {
    board = board || globalThis.gameState.board;
    for (const [dx, dy] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        let count = 1;
        for (const s of [1, -1]) {
            for (let i = 1; i < 5; i++) {
                const nx = x + s * i * dx, ny = y + s * i * dy;
                if (nx < 0 || nx >= 15 || ny < 0 || ny >= 15) break;
                if (board[nx][ny] !== player) break;
                count++;
            }
        }
        if (count >= 5) return true;
    }
    return false;
};

// ---- load AI modules in dependency order ----------------------------------
function loadSource(file) {
    const code = fs.readFileSync(file, 'utf8');
    (0, eval)(code);
}

// local-file fetch shim so MasterAI.loadModel works unmodified
globalThis.fetch = async function (url) {
    return {
        ok: true,
        status: 200,
        json: async () => JSON.parse(fs.readFileSync(url, 'utf8')),
    };
};

loadSource(path.join(__dirname, 'script_ai_extract.js'));  // levels 1-4
loadSource(path.join(FRONTEND, 'master_ai.js'));           // level 5
loadSource(path.join(FRONTEND, 'teacher_ai.js'));          // level 6
loadSource(path.join(FRONTEND, 'mcts_ai.js'));             // level 7

const MODEL_READY =
    MasterAI.loadModel(path.join(FRONTEND, 'master_model.json'));

async function pickMove(req) {
    await MODEL_READY;
    globalThis.gameState.board = req.board;
    const me = req.me;
    let move = null;
    switch (req.level) {
        case 1: move = basicAI(me); break;
        case 2: move = defenseAI(me); break;
        case 3: move = attackAI(me); break;
        case 4: move = expertAI(me); break;
        case 5:
            move = (typeof MasterAI !== 'undefined' && MasterAI.isLoaded())
                       ? MasterAI.pick(gameState.board, me)
                       : expertAI(me);
            break;
        case 6:
            move = TeacherAI.pick(gameState.board, me, {depth: 3, topK: 20});
            break;
        case 7:
            move = MctsAI.pick(gameState.board, me,
                               {simulations: 200, cPuct: 2.0, hybridAttack: 0.5});
            break;
        default:
            throw new Error('unknown level ' + req.level);
    }
    if (!move || typeof move.x !== 'number' || typeof move.y !== 'number') {
        throw new Error('level ' + req.level + ' returned no move');
    }
    return move;
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', (line) => {
    (async () => {
        try {
            const req = JSON.parse(line);
            const move = await pickMove(req);
            process.stdout.write(JSON.stringify(move) + '\n');
        } catch (err) {
            process.stdout.write(JSON.stringify({error: String(err)}) + '\n');
        }
    })();
});

process.stderr.write('[js_engine] ready, levels 1-7 loaded\n');
