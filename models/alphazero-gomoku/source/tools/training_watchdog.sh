#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

trainer_alive() {
  local pid cmd
  while read -r pid; do
    [ -r "/proc/${pid}/cmdline" ] || continue
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    case "${cmd}" in
      *"./bin/alphazero train "*) return 0 ;;
    esac
  done < <(pgrep -x alphazero || true)
  return 1
}

start_trainer() {
  printf '[%s] trainer missing; resuming from runtime checkpoint\n' "$(date -Is)" >> runtime/watchdog.log
  taskset -c 0-55 setsid nohup ./bin/alphazero train \
    --run-dir runtime \
    --workers 48 \
    --games-per-iter 80 \
    --sims 600 \
    --train-steps 200 \
    --batch 128 \
    --lr 0.001 \
    --wd 0.0001 \
    --value-weight 2 \
    --buffer 200000 \
    --max-moves 200 \
    --temp-moves 6 \
    --seed-hard-prob 0.3 \
    --cpuct 0.8 \
    --dir-eps 0.25 \
    --dir-alpha 0.3 \
    --fpu 0 \
    --gate-every 5 \
    --gate-games 20 \
    --gate-threshold 0.55 \
    --save-buffer-every 10 \
    --trunk 32 \
    --blocks 4 \
    --seed 42 \
    >> runtime.out 2>&1 < /dev/null &
  printf '[%s] trainer pid=%s\n' "$(date -Is)" "$!" >> runtime/watchdog.log
}

while true; do
  # Only the plateau monitor is allowed to create this sentinel, after all
  # three hard stop conditions pass.  Until then the trainer must stay alive.
  if [ -f runtime/PLATEAU_REACHED ]; then
    printf '[%s] plateau sentinel present; watchdog exits\n' "$(date -Is)" >> runtime/watchdog.log
    exit 0
  fi
  if ! trainer_alive; then
    start_trainer
  fi
  sleep 60
done
