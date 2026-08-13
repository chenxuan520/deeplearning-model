# Cloudflare Workers probe for xiangqi-8992

This model was probed on Cloudflare Workers and is not suitable for direct Worker inference.

Observed:

- Minimal Worker deployment: passed.
- CPU loop `n=200,000,000`: HTTP 200, about 2.25s client wall.
- CPU loop `n=500,000,000`: HTTP 503, `error code: 1102`.
- Bundling `steps_00010.model.bin` into the Worker: failed.
  - Wrangler dry run: `Total Upload: 18932.70 KiB / gzip: 13284.75 KiB`.
  - Cloudflare API rejected it: account Worker size limit `3 MiB`; paid limit commonly `10 MiB` is still below the gzipped model.
- R2 probe: blocked because R2 is not enabled on the account.
- Even with R2, the remaining problems are C++ to WASM, CPU time for 128-PUCT, no child process fork, and no reliable cross-request model/cache residency.

Conclusion: use Cloudflare Pages/Worker as static frontend or proxy only. Run native C++ inference on a separate backend.

Temporary probe Workers were deleted after testing.
