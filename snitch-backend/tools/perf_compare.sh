#!/bin/bash
# Verilator performance of one run_gemm.py configuration, the way
# many_gemms.sh measures it: run on Verilator (verify=true); run_gemm.py turns
# the DMA hart's trace into a CSV (gen_trace.py + extractDataFromJsons2.py)
# and prints its "dma" column, which this prints.
#
#   tools/perf_compare.sh [--driver <run_gemm.py-like script>] <MxNxKwm-n-k> \
#                         [xdsl|library] [remainder|pad] [-- <more run_gemm.py options>]
#
# Run from the host. --driver selects another driver script in snitch-backend/
# (e.g. a copy of run_gemm.py with different compiler settings, for
# before/after comparisons); give it a distinct Config.name so the two don't
# share an output directory. Only Verilator numbers are performance numbers:
# gvsoc is not cycle-accurate.
set -euo pipefail

SB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SB_ROOT"

driver=run_gemm.py
if [[ "${1:-}" == --driver ]]; then
  driver="$2"
  shift 2
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--driver script] MxNxKwm-n-k [xdsl|library] [remainder|pad] [-- options]" >&2
  exit 1
fi
shape=$1 mk=${2:-library} tiling=${3:-remainder}
shift $(( $# < 3 ? $# : 3 ))
[[ "${1:-}" == -- ]] && shift

# Forward run_gemm.py's opt-in experiment switches into the container, and OUT
# (a separate output directory for a "before" run of the same configuration).
env_args=()
for var in GEMM_STAGGER_N GEMM_PREFETCH GEMM_HW_BARRIER GEMM_K_REMAINDER SNITCH_FULL_TRACE OUT; do
  [[ -n "${!var:-}" ]] && env_args+=(-e "$var=${!var}")
done

label="$shape $mk $tiling"
if ! out=$(docker compose exec -T "${env_args[@]}" snitch "./$driver" "$shape" \
             --microkernel "$mk" --tiling "$tiling" --simulator verilator "$@" 2>&1) ||
   ! grep -q "^\[+\] verified" <<<"$out"; then
  echo "$label: FAILED"
  grep -E '^\[!\]|mismatch|error:' <<<"$out" | head -3
  exit 1
fi
echo "$label: verified, $(grep -oE 'dma=[0-9]+' <<<"$out")"
