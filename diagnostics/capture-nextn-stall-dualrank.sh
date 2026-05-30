#!/usr/bin/env bash
# capture-nextn-stall-dualrank.sh
#
# Dual-rank live capture for the cross-node TP=2 Qwen3.6-27B NEXTN decode-sync
# stall (the misnamed-#22511 "no response from detokenizer" health-check trip).
#
# WHY THIS EXISTS
# ---------------
# The 2026-05-28 on-death capture pinned the stall LOCUS from rank-0 ONLY:
#   - rank-0 NCCL watchdog C-stack: WorkNCCL(SeqNum=43216, OpType=BROADCAST,
#     NumelIn=8, PG ID 2) -> sample()@eagle_info_v2.py (the verify-sync
#     broadcast trio), ran 600037ms then collective-timeout-killed.
#   - rank-0 on-death py-spy: scheduler thread parked in
#     copy_done.synchronize() (process_batch_result_decode) -> why /health trips.
# rank-1's frame at the hung SeqNum was NEVER captured (its log had rolled, and
# node-2 has no py-spy installed). That single missing datum is what separates
# "the broadcast-fuse patch CURES the stall" from "it only mitigates it":
#   - if rank-1 is parked at the SAME broadcast (matching SeqNum) -> both ranks
#     reached a MATCHED collective that NCCL/transport failed to complete ->
#     fusion 3->1 does NOT fix it (look at NCCL/RDMA on the ConnectX link).
#   - if rank-1 is at a DIFFERENT collective / a later iteration / idle in
#     recv_requests -> a real cross-rank desync upstream of the trio -> the fix
#     is elsewhere (find the rank-asymmetric collective), fusion is partial.
#   - only if the evidence points specifically at an interleave WINDOW among the
#     three back-to-back broadcasts is the fuse the actual cure.
#
# This script catches BOTH ranks at the stall, before any log roll erases the
# evidence again. It does NOT reproduce the stall and does NOT touch the build.
#
# PREREQUISITES (the script checks and aborts if unmet)
#   - py-spy on BOTH nodes. node-2 currently lacks it; install once:
#       ssh "$RANK1_HOST" '~/.local/bin/uv tool install py-spy || pipx install py-spy'
#     (or rsync node-1's ~/.local/bin/py-spy over). py-spy needs to read another
#     process' memory: run under sudo OR ensure
#     /proc/sys/kernel/yama/ptrace_scope=0 on both nodes.
#   - The daily-driver TP=2 service is running (rank0 = $RANK0_HOST in tmux
#     session daily-driver-rank0; rank1 = $RANK1_HOST in daily-driver-rank1),
#     logging to /tmp/bench-results/daily-driver-rank{0,1}.log.
#
# USAGE
#   diagnostics/capture-nextn-stall-dualrank.sh
#     -> polls rank-0's log for the stall signal, then snapshots both ranks.
#   FORCE_NOW=1 diagnostics/capture-nextn-stall-dualrank.sh
#     -> skip the wait, snapshot both ranks immediately (use when you already
#        see the heartbeat frozen / a WARN line in the rank-0 log).
#
# Pairs with the diagnostic build commit (SGLANG_DEBUG_SPEC_DECODE_SYNC_WARN_SECS
# on branch gb10-spec-heartbeat-liveness): if that build is deployed with
# WARN_SECS>0, this script also triggers on its "[spec-decode-sync] ... py-spy
# BOTH ranks NOW" ERROR line, which fires ~WARN_SECS into the block instead of
# waiting the full health-check window.

set -uo pipefail

RANK0_HOST="${RANK0_HOST:?set RANK0_HOST to the rank-0 (TP0) node host/IP}"
RANK1_HOST="${RANK1_HOST:?set RANK1_HOST to the rank-1 (TP1) node host/IP}"
RANK0_LOG="${RANK0_LOG:-/tmp/bench-results/daily-driver-rank0.log}"
RANK1_LOG="${RANK1_LOG:-/tmp/bench-results/daily-driver-rank1.log}"
# Scheduler subprocess names: rank0 -> scheduler_0, rank1 -> scheduler_1.
RANK0_PROC_RE="${RANK0_PROC_RE:-scheduler_0}"
RANK1_PROC_RE="${RANK1_PROC_RE:-scheduler_1}"
PYSPY="${PYSPY:-py-spy}"
# Stall signals to watch for in the rank-0 log (any one trips the capture):
#  - the misnamed health-check line (covers the 20s and 60s variants), and
#  - the diagnostic build's explicit WARN line.
TRIGGER_RE="${TRIGGER_RE:-couldn.t get a response from detokenizer|\\[spec-decode-sync\\].*py-spy BOTH ranks NOW}"
POLL_SECS="${POLL_SECS:-2}"

TS="$(date '+%Y%m%d-%H%M%S')"
OUT="${OUT:-$HOME/notes/llm-research/wedge-captures-$TS}"
mkdir -p "$OUT"

log() { printf '[capture %s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

# --- preflight: py-spy on both nodes -----------------------------------------
preflight() {
  local ok=1
  if ! command -v "$PYSPY" >/dev/null 2>&1; then
    log "ABORT: py-spy not found locally (rank-0 host). Install it on $RANK0_HOST."
    ok=0
  fi
  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$RANK1_HOST" "command -v $PYSPY >/dev/null 2>&1"; then
    log "ABORT: py-spy not found on $RANK1_HOST (rank-1). This is the gap that"
    log "       lost rank-1 last time. Install first:"
    log "         ssh $RANK1_HOST '~/.local/bin/uv tool install py-spy || pipx install py-spy'"
    ok=0
  fi
  [ "$ok" = 1 ] || exit 2
}

# --- pid discovery (newest matching scheduler proc) --------------------------
pid_local()  { pgrep -nf "$RANK0_PROC_RE"; }
pid_remote() { ssh -o ConnectTimeout=5 -o BatchMode=yes "$RANK1_HOST" "pgrep -nf '$RANK1_PROC_RE'"; }

# --- one rank's snapshot: py-spy dump (python+native) + a 30s py-spy record ---
snap_rank0() {
  local pid; pid="$(pid_local)"
  if [ -z "${pid:-}" ]; then log "WARN: no rank-0 scheduler pid [$RANK0_PROC_RE]"; return; fi
  log "rank-0 pid=$pid -> dump (python+native)"
  # --native gives the C/NCCL frames (where the broadcast/synchronize actually sits)
  sudo -n "$PYSPY" dump --pid "$pid" --native    > "$OUT/rank0-pyspy-dump-native.txt" 2>&1 \
    || "$PYSPY" dump --pid "$pid" --native        > "$OUT/rank0-pyspy-dump-native.txt" 2>&1
  sudo -n "$PYSPY" dump --pid "$pid"              > "$OUT/rank0-pyspy-dump.txt"        2>&1 \
    || "$PYSPY" dump --pid "$pid"                  > "$OUT/rank0-pyspy-dump.txt"        2>&1
  # A short record confirms the stack is STUCK (not just sampled at one instant).
  sudo -n "$PYSPY" record --pid "$pid" --duration 30 --rate 5 --format raw \
        --output "$OUT/rank0-pyspy-record.txt" 2>/dev/null \
    || "$PYSPY" record --pid "$pid" --duration 30 --rate 5 --format raw \
        --output "$OUT/rank0-pyspy-record.txt" 2>/dev/null &
}

snap_rank1() {
  local pid; pid="$(pid_remote)"
  if [ -z "${pid:-}" ]; then log "WARN: no rank-1 scheduler pid on $RANK1_HOST [$RANK1_PROC_RE]"; return; fi
  log "rank-1 pid=$pid on $RANK1_HOST -> dump (python+native)"
  ssh -o ConnectTimeout=5 "$RANK1_HOST" \
      "sudo -n $PYSPY dump --pid $pid --native || $PYSPY dump --pid $pid --native" \
      > "$OUT/rank1-pyspy-dump-native.txt" 2>&1
  ssh -o ConnectTimeout=5 "$RANK1_HOST" \
      "sudo -n $PYSPY dump --pid $pid || $PYSPY dump --pid $pid" \
      > "$OUT/rank1-pyspy-dump.txt" 2>&1
  ssh -o ConnectTimeout=5 "$RANK1_HOST" \
      "sudo -n $PYSPY record --pid $pid --duration 30 --rate 5 --format raw --output /tmp/rank1-pyspy-record.txt 2>/dev/null || \
       $PYSPY record --pid $pid --duration 30 --rate 5 --format raw --output /tmp/rank1-pyspy-record.txt 2>/dev/null" &
}

# --- snapshot logs + NCCL identity NOW (beat the log roll) --------------------
snap_context() {
  log "snapshotting both rank logs + NCCL identity (before any roll)"
  cp -f "$RANK0_LOG" "$OUT/rank0-full.log" 2>/dev/null || log "WARN: could not copy $RANK0_LOG"
  scp -o ConnectTimeout=5 "$RANK1_HOST:$RANK1_LOG" "$OUT/rank1-full.log" 2>/dev/null \
    || log "WARN: could not scp $RANK1_LOG from $RANK1_HOST"
  # Confirm BOTH ranks resolve the SAME libnccl (a reverted/partial swap would
  # itself produce a 'matched collective never completes' signature -> fusion
  # would NOT fix it; rule it in/out here per feedback_wheel_nccl_lacks_sm121).
  {
    echo "== rank0 libnccl ($RANK0_HOST) =="
    ls -l "$HOME/projects/sglang-server/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/"libnccl.so* 2>/dev/null
    md5sum "$HOME/projects/sglang-server/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2" 2>/dev/null
    echo "== rank1 libnccl ($RANK1_HOST) =="
    ssh -o ConnectTimeout=5 "$RANK1_HOST" \
      "ls -l \$HOME/projects/sglang-server/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so*; \
       md5sum \$HOME/projects/sglang-server/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2" 2>/dev/null
  } > "$OUT/libnccl-identity.txt" 2>&1
}

# --- main --------------------------------------------------------------------
preflight
log "output dir: $OUT"

if [ "${FORCE_NOW:-0}" != "1" ]; then
  log "waiting for stall signal in $RANK0_LOG (regex: $TRIGGER_RE)"
  log "  (set FORCE_NOW=1 to snapshot immediately)"
  # Block until a NEW matching line appears, then proceed. tail -F survives roll.
  ( tail -n0 -F "$RANK0_LOG" 2>/dev/null & echo $! >"$OUT/.tailpid" ) | \
    grep -m1 -E "$TRIGGER_RE" >/dev/null
  kill "$(cat "$OUT/.tailpid" 2>/dev/null)" 2>/dev/null; rm -f "$OUT/.tailpid"
  log "stall signal seen -> capturing BOTH ranks NOW"
fi

# Order matters: snapshot context first (logs roll fastest), then both ranks'
# stacks as close together in time as possible.
snap_context
snap_rank0
snap_rank1
wait   # let the two 30s `record`s finish

# Pull rank-1's record artifact back next to the others.
scp -o ConnectTimeout=5 "$RANK1_HOST:/tmp/rank1-pyspy-record.txt" \
    "$OUT/rank1-pyspy-record.txt" 2>/dev/null || true

log "DONE. Decisive datum: compare rank0 vs rank1 --native dumps."
log "  - rank1 at the SAME broadcast SeqNum as rank0  -> matched-collective"
log "    non-completion: NCCL/RDMA transport, NOT the fuse. Check libnccl-identity.txt."
log "  - rank1 at a DIFFERENT collective / later iter / idle in recv_requests"
log "    -> real cross-rank desync UPSTREAM of the trio; fix is elsewhere."
log "  - evidence of an interleave window across the 3 back-to-back broadcasts"
log "    -> the broadcast-fuse patch (gb10-spec-verify-broadcast-fuse) is the cure."
log "Artifacts in: $OUT"
