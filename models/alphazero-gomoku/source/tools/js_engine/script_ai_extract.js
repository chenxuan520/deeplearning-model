function expertAI(me) {
    let bestMove = { x: -1, y: -1 };
    let bestValue = -Infinity;
    const aiPlayer = me || 2;
    const humanPlayer = 3 - aiPlayer;
    const depth = 2; // Search depth. Higher is smarter but slower. 2 is a good balance.

    const possibleMoves = getPossibleMoves(gameState.board);

    // If board is empty, play in the center
    if (possibleMoves.length === BOARD_SIZE * BOARD_SIZE) {
        return { x: Math.floor(BOARD_SIZE / 2), y: Math.floor(BOARD_SIZE / 2) };
    }

    for (let i = 0; i < possibleMoves.length; i++) {
        const move = possibleMoves[i];
        gameState.board[move.x][move.y] = aiPlayer;
        let moveValue = minimax(gameState.board, depth, -Infinity, Infinity, false, humanPlayer, aiPlayer);
        gameState.board[move.x][move.y] = 0; // Backtrack

        if (moveValue > bestValue) {
            bestValue = moveValue;
            bestMove = move;
        }
    }
    console.log("Expert AI chooses move:", bestMove, "with score:", bestValue);
    return bestMove;
}

function minimax(board, depth, alpha, beta, isMaximizingPlayer, humanPlayer, aiPlayer) {
    if (depth === 0 || isGameOver(board)) {
        return evaluateBoard(board, humanPlayer, aiPlayer);
    }

    const possibleMoves = getPossibleMoves(board);

    if (isMaximizingPlayer) {
        let maxEval = -Infinity;
        for (const move of possibleMoves) {
            board[move.x][move.y] = aiPlayer;
            let an_eval = minimax(board, depth - 1, alpha, beta, false, humanPlayer, aiPlayer);
            board[move.x][move.y] = 0;
            maxEval = Math.max(maxEval, an_eval);
            alpha = Math.max(alpha, an_eval);
            if (beta <= alpha) {
                break;
            }
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (const move of possibleMoves) {
            board[move.x][move.y] = humanPlayer;
            let an_eval = minimax(board, depth - 1, alpha, beta, true, humanPlayer, aiPlayer);
            board[move.x][move.y] = 0;
            minEval = Math.min(minEval, an_eval);
            beta = Math.min(beta, an_eval);
            if (beta <= alpha) {
                break;
            }
        }
        return minEval;
    }
}

function getPossibleMoves(board) {
    const moves = [];
    const R = 2; // Radius to check around existing stones

    let hasStones = false;
    for(let i=0; i<BOARD_SIZE; i++) {
        for(let j=0; j<BOARD_SIZE; j++) {
            if(board[i][j] !== 0) {
                hasStones = true;
                break;
            }
        }
        if(hasStones) break;
    }

    if (!hasStones) {
        moves.push({ x: Math.floor(BOARD_SIZE / 2), y: Math.floor(BOARD_SIZE / 2) });
        return moves;
    }

    const candidates = new Set();
    for (let i = 0; i < BOARD_SIZE; i++) {
        for (let j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] === 0) {
                // Check if there is any stone nearby
                for (let dx = -R; dx <= R; dx++) {
                    for (let dy = -R; dy <= R; dy++) {
                        if (dx === 0 && dy === 0) continue;
                        const x = i + dx;
                        const y = j + dy;
                        if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE && board[x][y] !== 0) {
                            candidates.add(`${i},${j}`);
                            break; // Found a neighbor, no need to check other neighbors for this empty spot
                        }
                    }
                     if(candidates.has(`${i},${j}`)) break;
                }
            }
        }
    }

    candidates.forEach(c => {
        const [x, y] = c.split(',').map(Number);
        moves.push({x, y});
    });
    return moves;
}

function evaluateBoard(board, humanPlayer, aiPlayer) {
    let score = 0;
    score += evaluateDirection(board, 1, 0, humanPlayer, aiPlayer); // Horizontal
    score += evaluateDirection(board, 0, 1, humanPlayer, aiPlayer); // Vertical
    score += evaluateDirection(board, 1, 1, humanPlayer, aiPlayer); // Diagonal \
    score += evaluateDirection(board, 1, -1, humanPlayer, aiPlayer); // Anti-diagonal /
    return score;
}

function evaluateDirection(board, dx, dy, humanPlayer, aiPlayer) {
    let score = 0;
    for (let i = 0; i < BOARD_SIZE; i++) {
        for (let j = 0; j < BOARD_SIZE; j++) {
            let line = [];
            let valid = true;
            for (let k = 0; k < 5; k++) {
                const x = i + k * dx;
                const y = j + k * dy;
                if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) {
                    line.push(board[x][y]);
                } else {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                score += scoreLine(line, humanPlayer, aiPlayer);
            }
        }
    }
    return score;
}

function scoreLine(line, humanPlayer, aiPlayer) {
    let aiCount = 0;
    let humanCount = 0;

    for (const cell of line) {
        if (cell === aiPlayer) aiCount++;
        else if (cell === humanPlayer) humanCount++;
    }

    if (aiCount > 0 && humanCount > 0) return 0; // Mixed line has no potential

    if (aiCount === 5) return 100000;
    if (humanCount === 5) return -1000000; // A definite loss is worse than a potential win
    if (aiCount === 4) return 10000;
    if (humanCount === 4) return -50000;
    if (aiCount === 3) return 1000;
    if (humanCount === 3) return -5000;
    if (aiCount === 2) return 100;
    if (humanCount === 2) return -500;
    if (aiCount === 1) return 10;
    if (humanCount === 1) return -50;

    return 0;
}

function isGameOver(board) {
    for (let i = 0; i < BOARD_SIZE; i++) {
        for (let j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] !== 0) {
                if (checkWin(i, j, board[i][j], board)) return true;
            }
        }
    }
    return false;
}




// =======================================================
// ========== HEURISTIC AI (BASIC/DEFENSE/ATTACK) ========
// =======================================================
// 三档难度共用同一份打分函数, 通过 AI_PROFILES 配置不同权重.
// 评分方式: 对每个空位, 在 4 条线 (横/竖/\/) 上向两端扫描, 累加
//   (a) 防御得分 = 对手连子越多越值得堵;
//   (b) 进攻得分 = 己方连子越多越值得延伸.
// 来源: 原 C++ computer / computerTry / computerThree 三个函数, 已合并去重.
// 行为基本等价, 仅做了两处清理:
//   1) 边界判断统一为 [0, N), 修掉原版几处不对称的 off-by-one;
//   2) 打分相同的候选改为均匀随机 (原版第一个发现的候选会被双倍计数).

const AI_PROFILES = {
    basic: {
        def: { hvFactor: 5, diaFactor: 7, t2: 30, t3: 50, blockedOwn: -30, emptyBonus: 0 },
        atk: { factor: 8,  t2: 20, t3: 100, blockedOpp: -15, blockedOppQ: 0,
               emptyBonus: 0, edgePenalty: 0,   skipBonus: 0, openBonus: 0  },
    },
    defense: {
        def: { hvFactor: 5, diaFactor: 7, t2: 30, t3: 50, blockedOwn: -30, emptyBonus: 3 },
        atk: { factor: 8,  t2: 20, t3: 100, blockedOpp: -15, blockedOppQ: 0,
               emptyBonus: 4, edgePenalty: 0,   skipBonus: 0, openBonus: 0  },
    },
    attack: {
        def: { hvFactor: 6, diaFactor: 6, t2: 40, t3: 50, blockedOwn: -40, emptyBonus: 3 },
        atk: { factor: 8,  t2: 40, t3: 120, blockedOpp: -30, blockedOppQ: -35,
               emptyBonus: 4, edgePenalty: -15, skipBonus: 5, openBonus: 30 },
    },
};

const DIRS = [
    { dx: 1, dy:  0, diag: false }, // 横
    { dx: 0, dy:  1, diag: false }, // 竖
    { dx: 1, dy:  1, diag: true  }, // \
    { dx: 1, dy: -1, diag: true  }, // /
];

function inBoard(x, y) {
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
}

// 给单个空位打分: 越大越值得落子.
function scoreCellHeuristic(board, x, y, me, it, cfg) {
    let value = 0;

    // ----- 防御扫描: 数 (x,y) 周围的对手棋子 -----
    for (const { dx, dy, diag } of DIRS) {
        const factor = diag ? cfg.def.diaFactor : cfg.def.hvFactor;
        let t = 0; // 该轴上累积的对手连子数
        for (const sign of [1, -1]) {
            for (let s = 1; s < BOARD_SIZE; s++) {
                const nx = x + s * sign * dx;
                const ny = y + s * sign * dy;
                if (!inBoard(nx, ny)) break;
                const c = board[nx][ny];
                if (c === it) {
                    value += 5 + factor * t;
                    if (t >= 2) value += cfg.def.t2;
                    if (t >= 3) value += cfg.def.t3;
                    t++;
                } else {
                    if (c === me && t >= 2) value += cfg.def.blockedOwn;
                    if (c === 0)           value += cfg.def.emptyBonus;
                    break;
                }
            }
        }
    }

    // ----- 进攻扫描: 数 (x,y) 周围的己方棋子 -----
    for (const { dx, dy } of DIRS) {
        let t = 0, q = 0; // t = 己方连子数, q = 同轴被对手截断的次数
        for (const sign of [1, -1]) {
            // 边界惩罚: 紧邻方向已出界, 这条线被棋盘边卡死
            if (cfg.atk.edgePenalty && !inBoard(x + sign * dx, y + sign * dy)) {
                value += cfg.atk.edgePenalty;
            }
            for (let s = 1; s < BOARD_SIZE; s++) {
                const nx = x + s * sign * dx;
                const ny = y + s * sign * dy;
                if (!inBoard(nx, ny)) break;
                const c = board[nx][ny];
                if (c === me) {
                    value += 6 + cfg.atk.factor * t;
                    if (t >= 2) value += cfg.atk.t2;
                    if (t >= 3) value += cfg.atk.t3;
                    t++;
                } else {
                    if (c === it && t >= 2) {
                        value += cfg.atk.blockedOpp;
                        if (cfg.atk.blockedOppQ !== 0) {
                            q++;
                            if (q > 1) value += cfg.atk.blockedOppQ;
                        }
                    }
                    if (c === 0) {
                        value += cfg.atk.emptyBonus;
                        // 跳子加分: 落子之后再跟着 me 然后非对手 -> 跳跃式连子潜力
                        if (cfg.atk.skipBonus) {
                            const nx2 = nx +     sign * dx;
                            const ny2 = ny +     sign * dy;
                            const nx3 = nx + 2 * sign * dx;
                            const ny3 = ny + 2 * sign * dy;
                            if (inBoard(nx3, ny3) &&
                                board[nx2][ny2] === me &&
                                board[nx3][ny3] !== it) {
                                value += cfg.atk.skipBonus;
                            }
                        }
                    }
                    break;
                }
            }
        }
        // 该轴未被对手截断 + 自己已经连了 2+ -> 活三/活四潜力
        if (cfg.atk.openBonus && q === 0 && t >= 2) value += cfg.atk.openBonus;
    }

    return value;
}

// 扫全盘, 取分数最高的空位 (并列均匀随机).
function heuristicAI(profile, me) {
    me = me || 2;
    const it = 3 - me;
    const board = gameState.board;
    const cfg = AI_PROFILES[profile];

    // 空棋盘: 落天元
    let hasStones = false;
    for (let i = 0; i < BOARD_SIZE && !hasStones; i++) {
        for (let j = 0; j < BOARD_SIZE && !hasStones; j++) {
            if (board[i][j] !== 0) hasStones = true;
        }
    }
    if (!hasStones) {
        return { x: Math.floor(BOARD_SIZE / 2), y: Math.floor(BOARD_SIZE / 2) };
    }

    let bestValue = -Infinity;
    let candidates = [];
    for (let x = 0; x < BOARD_SIZE; x++) {
        for (let y = 0; y < BOARD_SIZE; y++) {
            if (board[x][y] !== 0) continue;
            const v = scoreCellHeuristic(board, x, y, me, it, cfg);
            if (v > bestValue) {
                bestValue = v;
                candidates = [{ x, y }];
            } else if (v === bestValue && candidates.length < 99) {
                candidates.push({ x, y });
            }
        }
    }

    if (candidates.length === 0) return null; // 兜底, 上面已经处理过空盘
    return candidates[Math.floor(Math.random() * candidates.length)];
}

function basicAI(me)   { return heuristicAI('basic',   me); }
function defenseAI(me) { return heuristicAI('defense', me); }
function attackAI(me)  { return heuristicAI('attack',  me); }
