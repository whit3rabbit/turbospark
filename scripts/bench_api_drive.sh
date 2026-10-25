#!/bin/bash
# One server at a time (20 GB + 15 GB models do not co-reside in 36 GB).
# Arm order is rotated each round so thermal drift does not favor an engine.
#   BENCH_DIR          where logs and result JSON go (default target/bench-api)
#   TURBOSPARK_SERVER  the turbospark-server binary (default target/release/turbospark-server)
#   TURBOSPARK_MODEL   the .gturbo install for the ts-* arms (default ~/models/qwen38-27b-mtp.gturbo)
#   KAPITSA_TOKENIZER  tokenizer.json used to count tokens for every arm
HERE=$(cd "$(dirname "$0")" && pwd)
S=${BENCH_DIR:-$HERE/../target/bench-api}
mkdir -p "$S"
TS=${TURBOSPARK_SERVER:-$HERE/../target/release/turbospark-server}
MODEL=${TURBOSPARK_MODEL:-$HOME/models/qwen38-27b-mtp.gturbo}
TOK=${KAPITSA_TOKENIZER:-$HOME/.cache/kapitsa/models/qwen3.8-27b-4bit/tokenizer.json}
PORT=8090
CASES=short-explanation,medium-review,copy-heavy
RUNS=3
SRV=

stop() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=; }
trap stop EXIT

start_arm() {
  case $1 in
    ts-mtp)  $TS --model $MODEL --port $PORT --speculative auto > $S/srv-$1.log 2>&1 & ;;
    ts-seq)  $TS --model $MODEL --port $PORT --speculative off > $S/srv-$1.log 2>&1 & ;;
    k-kflash) kapitsa serve --model 27b --port $PORT > $S/srv-$1.log 2>&1 & ;;
    k-mtp)   kapitsa serve --model 27b --port $PORT --dflash none > $S/srv-$1.log 2>&1 & ;;
  esac
  SRV=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "server for $1 died"; return 1; }
    sleep 2
  done
  echo "server for $1 not ready"; return 1
}

run_arm() {
  local arm=$1 round=$2
  for i in $(seq 1 60); do
    l=$(uptime | sed 's/.*averages: //' | awk '{print $1}')
    awk "BEGIN{exit !($l < 2.5)}" && break
    sleep 5
  done
  echo "### round $round arm $arm load1=$l swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
  start_arm $arm || { stop; return; }
  python3 -I "$HERE/bench_api.py" --base-url http://127.0.0.1:$PORT --model m --tokenizer $TOK \
     --label $arm --cases $CASES --runs $RUNS --out $S/res-$round-$arm.json
  stop
  sleep 5
}

for arm in ts-mtp k-kflash ts-seq k-mtp; do run_arm $arm 1; done
for arm in k-mtp ts-seq k-kflash ts-mtp; do run_arm $arm 2; done
echo ALLDONE
