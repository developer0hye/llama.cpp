#!/bin/bash
# DSV4.1-Flash A/B: base (0ac1a7fd3) vs tile-only (91d6c2a41) vs both (3113b8623), sm_86.
# Usage (from a clone of this branch):
#   MODEL=/path/to/model.gguf bash dsv41-ab.sh [step...]
#   steps: build test unit e2e kl report (default: all; build/test/unit do not need MODEL)
# Optional env: WORK (default ~/dsv41-ab), GPU (free GPU for test/unit, default 0),
#   EXTRA (extra flags for llama-bench/batched-bench/perplexity, e.g. "-fa on"), JOBS (build jobs),
#   UNIT_PAT (test-backend-ops perf case filter).
set -euo pipefail

SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$HOME/dsv41-ab}
GPU=${GPU:-0}
EXTRA=${EXTRA:-}
JOBS=${JOBS:-$(nproc)}
UNIT_PAT=${UNIT_PAT:-n_mats=384,n_used=6,b=0,m=(2304|5120),}
BASE=0ac1a7fd3
TILE=91d6c2a41
NEW=3113b8623
STEPS=${*:-build test unit e2e kl report}
RES=$WORK/results
mkdir -p "$WORK" "$RES"

machine() { # record load next to every measurement
  { echo "# $(date) $1"; uptime; nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,utilization.gpu,memory.used --format=csv,noheader; } >> "$RES/machine.txt"
}

need_model() { [ -n "${MODEL:-}" ] && [ -f "$MODEL" ] || { echo "set MODEL=/path/to/model.gguf"; exit 1; }; }

do_build() {
  command -v nvcc >/dev/null || { echo "nvcc not in PATH"; exit 1; }
  [ -d "$WORK/new" ]  || git -C "$SRC" worktree add -q --detach "$WORK/new"  $NEW
  [ -d "$WORK/tile" ] || git -C "$SRC" worktree add -q --detach "$WORK/tile" $TILE
  if [ ! -d "$WORK/base" ]; then
    git -C "$SRC" worktree add -q --detach "$WORK/base" $BASE
    git -C "$WORK/base" checkout -q $NEW -- tests/test-backend-ops.cpp # same perf cases in every arm
  fi
  for t in base tile new; do
    echo "== build $t ($(git -C "$WORK/$t" log -1 --format=%h))"
    cmake -S "$WORK/$t" -B "$WORK/$t/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release > "$RES/cmake_$t.log"
    cmake --build "$WORK/$t/build" -j "$JOBS" --target test-backend-ops llama-bench llama-batched-bench llama-perplexity > "$RES/build_$t.log"
  done
}

do_test() {
  for op in MUL_MAT_ID MUL_MAT_ID_FUSION MUL_MAT_ID_SHARED MUL_MAT; do
    CUDA_VISIBLE_DEVICES=$GPU "$WORK/new/build/bin/test-backend-ops" test -b CUDA0 -o $op > "$RES/test_$op.txt" 2>&1 || true
    echo "$op: $(grep -E 'tests passed|FAIL' "$RES/test_$op.txt" | tail -1)"
  done
}

do_unit() { # arms base,tile,new,base on one GPU
  local i=0
  for t in base tile new base; do
    machine "unit arm$i $t"
    CUDA_VISIBLE_DEVICES=$GPU "$WORK/$t/build/bin/test-backend-ops" perf -b CUDA0 -o MUL_MAT_ID,MUL_MAT_ID_FUSION -p "$UNIT_PAT" > "$RES/unit_${i}_$t.txt" 2>&1
    i=$((i+1))
  done
}

do_e2e() { # full model, all GPUs, arms base,tile,new,base
  need_model
  local i=0
  for t in base tile new base; do
    machine "e2e arm$i $t"
    "$WORK/$t/build/bin/llama-bench" -m "$MODEL" -ngl 99 -ub 512 -p 8,16,32,64,128,256,512,2048 -n 128 -r 3 -o md $EXTRA > "$RES/bench_${i}_$t.md" 2> "$RES/bench_${i}_$t.log"
    "$WORK/$t/build/bin/llama-batched-bench" -m "$MODEL" -ngl 99 -c 8192 -ub 512 -npp 128 -ntg 64 -npl 1,4,8,16,32 $EXTRA > "$RES/bb_${i}_$t.txt" 2>&1
    i=$((i+1))
  done
}

do_kl() { # output equality: base vs new logits at the same ub, 512 (MMQ path) and 16 (MMVQ path)
  need_model
  local txt=$RES/kl_text.txt
  cat "$SRC"/docs/*.md "$SRC"/docs/*/*.md > "$txt"
  machine "kl"
  for ub in 512 16; do
    "$WORK/base/build/bin/llama-perplexity" -m "$MODEL" -ngl 99 -f "$txt" -c 512 --chunks 8 -ub $ub $EXTRA --kl-divergence-base "$WORK/kl_base_ub$ub.bin" > "$RES/kl_base_ub$ub.txt" 2>&1
    "$WORK/new/build/bin/llama-perplexity" -m "$MODEL" -ngl 99 -f "$txt" -c 512 --chunks 8 -ub $ub $EXTRA --kl-divergence-base "$WORK/kl_base_ub$ub.bin" --kl-divergence > "$RES/kl_new_ub$ub.txt" 2>&1
    echo "ub=$ub: $(grep -E 'Mean +KLD' "$RES/kl_new_ub$ub.txt" | tail -1)"
  done
}

# input lines "arm key value"; base = mean of arm 0 and arm 3
RATIOS='{ k = $2; v[k, $1] = $3; if (!(k in seen)) { seen[k] = 1; ord[++n] = k } }
END {
  print hdr " | base | tile | new | tile/base | new/base"
  for (j = 1; j <= n; j++) {
    k = ord[j]; b = (v[k, 0] + v[k, 3]) / 2
    if (b > 0) printf "%s | %.1f | %.1f | %.1f | %.2f | %.2f\n", k, b, v[k, 1], v[k, 2], v[k, 1] / b, v[k, 2] / b
  }
}'

do_report() {
  local out=$RES/summary.txt
  {
    echo "## commits: base $BASE, tile $TILE, new $NEW; EXTRA='$EXTRA' MODEL='${MODEL:-}'"
    echo "## machine"; cat "$RES/machine.txt"
    echo "## test"
    for f in "$RES"/test_*.txt; do [ -f "$f" ] && echo "$(basename "$f" .txt): $(grep -E 'tests passed' "$f" | tail -1)"; done
    if ls "$RES"/unit_0_*.txt > /dev/null 2>&1; then
      echo "## unit: us/run (lower is better)"
      for i in 0 1 2 3; do
        awk -v a=$i '/us\/run/ { k = $1; sub(/:$/, "", k); gsub(/type_b=f32,|,b=0|,m_v=0|,ids_offset=0|,skip=0/, "", k); for (f = 1; f <= NF; f++) if ($f == "us/run") print a, k, $(f - 1) }' "$RES"/unit_${i}_*.txt
      done | awk -v hdr=case "$RATIOS"
    fi
    if ls "$RES"/bench_0_*.md > /dev/null 2>&1; then
      echo "## llama-bench: t/s (higher is better)"
      for i in 0 1 2 3; do
        awk -F'|' -v a=$i '$0 ~ /^\|/ && $(NF-2) ~ /(pp|tg)[0-9]/ { k = $(NF-2); gsub(/ /, "", k); split($(NF-1), p, " "); print a, k, p[1] }' "$RES"/bench_${i}_*.md
      done | awk -v hdr=test "$RATIOS"
      echo "## batched-bench: decode t/s S_TG, prompt 128 per sequence (higher is better)"
      for i in 0 1 2 3; do
        awk -F'|' -v a=$i '$0 ~ /^\| +[0-9]/ { b = $4; gsub(/ /, "", b); s = $9; gsub(/ /, "", s); print a, "npl=" b, s }' "$RES"/bb_${i}_*.txt
      done | awk -v hdr=npl "$RATIOS"
    fi
    for f in "$RES"/kl_new_ub*.txt; do [ -f "$f" ] && { echo "## $(basename "$f" .txt)"; grep -E 'Mean +KLD|Maximum KLD|Same top p' "$f"; }; done
  } > "$out"
  cat "$out"
  tar czf "$WORK/dsv41-ab-results.tgz" -C "$WORK" results
  echo "-> $out, $WORK/dsv41-ab-results.tgz"
}

machine "start: $STEPS"
for s in $STEPS; do echo "### $s"; do_$s; done
