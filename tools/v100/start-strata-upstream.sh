#!/usr/bin/env bash
# Strata upstream (official Niko1221/Strata, engine 0.1.63) on 2x V100-SXM2-16GB.
#
#   start-strata-upstream.sh start [config]   # config: strata-upstream-best | strata-upstream-safe
#   start-strata-upstream.sh stop
#   start-strata-upstream.sh status
#
# The engine binary comes from the upstream-sync branch of this repo:
#   ~/strata-upstream-sync/build/strata          (worktree of branch upstream-sync-20261121)
# Build recipe (official docs/NVIDIA_V100.md + SOURCE_INSTALL.md + OLDER_GPUS.md):
#   cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
#         -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
#   cmake --build build --target strata -j 24
#
# Every number this script relies on was measured on this box; see
# docs/HANDOFF-V100-UPSTREAM-SYNC.md.
set -uo pipefail

INSTALL_DIR="$HOME/.local/share/strata/install"
LOG_DIR="/data/models/logs"
PORT=8080

pick_terminal() {
  local base="$HOME/.local/share/strata/terminal"
  # SERVER_DIR is the terminal root: the script runs "serve/server.py" relative to it
  if [ -x "$base/current/python/bin/python3" ]; then
    TERMINAL_PY="$base/current/python/bin/python3"
    SERVER_DIR="$base/current"
  else
    TERMINAL_PY="$base/0.1.49.2/python/bin/python3"
    SERVER_DIR="$base/0.1.49.2"
  fi
}

cfg_value() {  # cfg_value <config-path> <key>
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' "$1" "$2" 2>/dev/null
}

do_start() {
  local name="${1:-strata-upstream-best}"
  local cfg="$INSTALL_DIR/$name.json"
  [ -f "$cfg" ] || { echo "no such config: $cfg" >&2; exit 1; }

  if ss -lnt 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is busy - stop the other service first (strata / ninfer / ...)" >&2
    exit 1
  fi

  # RLIMIT_MEMLOCK: the 26.8 GiB PLE table has to be locked in RAM, otherwise the engine
  # can only lock 15.7 GiB and loses ~6-10% of decode speed (it says so itself).
  local limit; limit=$(ulimit -l)
  if [ "$limit" != "unlimited" ]; then
    if sudo -n prlimit --memlock=unlimited --pid $$ 2>/dev/null; then
      echo "memlock: raised to unlimited"
    else
      echo "WARNING: RLIMIT_MEMLOCK is ${limit} KB, so the PLE table stays unlocked." >&2
      echo "         decode loses ~6-10%. The permanent fix is already installed on this box:" >&2
      echo "           /etc/security/limits.d/99-memlock.conf   (chenjx - memlock unlimited;" >&2
      echo "           it applies to every NEW login session - log out and back in)" >&2
      echo "         or run this service from a systemd unit with LimitMEMLOCK=infinity." >&2
    fi
  fi

  pick_terminal
  mkdir -p "$LOG_DIR"
  local log="$LOG_DIR/strata-upstream-$(date -u +%Y%m%d-%H%M%S).log"
  local key; key=$(cfg_value "$cfg" api_key)

  cd "$SERVER_DIR" || exit 1
  setsid nohup "$TERMINAL_PY" serve/server.py --engine strata --config "$cfg" \
      --port "$PORT" ${key:+--api-key "$key"} > "$log" 2>&1 < /dev/null &
  echo "log: $log"

  local i
  for i in $(seq 1 60); do
    sleep 10
    grep -q '^ready:' "$log" 2>/dev/null && break
    if ! pgrep -f 'serve/serv[e]r.py' >/dev/null; then
      echo "the server died:" >&2; tail -20 "$log" >&2; exit 1
    fi
  done
  grep -q '^ready:' "$log" || { echo "not ready after 10 minutes, see $log" >&2; exit 1; }
  grep -m1 '^ready:' "$log"

  # the lines worth looking at after every start
  local elog; elog=$(cfg_value "$cfg" log)
  [ -n "$elog" ] && grep -E 'PLE table.*locked|expert cache auto|layer split across|pcie_frac' "$elog" | tail -4
}

do_stop() {
  # safe stop: match the terminal python, never pkill -f a pattern that also matches this shell
  local pid exe
  for pid in $(pgrep -f 'serve/serv[e]r.py' 2>/dev/null); do
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    case "$exe" in
      *"/terminal/0.1.49.2/python/bin/python3"*|*"/terminal/current/python/bin/python3"*) kill "$pid" ;;
    esac
  done
  for i in $(seq 1 30); do ss -lnt 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
  ss -lnt 2>/dev/null | grep -q ":$PORT " && echo "port $PORT still busy" || echo "port $PORT free"
}

do_status() {
  echo "== process =="
  ps -eo pid,etime,args | grep '[s]erve/server[.]py' || echo "not running"
  echo "== api =="
  curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" -H 'Authorization: Bearer chenjx-v100' | head -c 300; echo
  echo "== gpu =="
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
}

case "${1:-start}" in
  start)  do_start "${2:-strata-upstream-best}" ;;
  stop)   do_stop ;;
  status) do_status ;;
  *) echo "usage: $0 {start [config]|stop|status}" >&2; exit 2 ;;
esac
