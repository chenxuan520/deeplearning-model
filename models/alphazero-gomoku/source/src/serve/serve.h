#pragma once

#include <string>

namespace az {

// Tiny blocking HTTP server exposing the trained model as a move API.
//   POST /api/move  body {"board":[[0/1/2]x15x15],"me":1|2} -> {"x":r,"y":c}
//   GET  /health    -> {"ok":true}
// CORS is wide open so any static frontend can call it.
// single-process: one request handled at a time (a browser game needs only
// sequential moves anyway).
int ServeMoves(const std::string &model_path, int port, int sims,
               int worker_threads);

} // namespace az
