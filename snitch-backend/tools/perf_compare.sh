#!/bin/bash
# Verilator performance of one run_gemm.py configuration, the way
# many_gemms.sh measures it: run on Verilator (verify=true), turn the DMA
# hart's trace into a CSV with extract_dma_csv.sh, print its "dma" column.
#
#   tools/perf_compare.sh [--driver <run_gemm.py-like script>] N M K n m k [microkernel]
#
# Run from the host (extract_dma_csv.sh docker-execs into the dev container).
# --driver selects another driver script in snitch-backend/ (e.g. a copy of
# run_gemm.py with different compiler settings and a distinct output-directory
# suffix, for before/after comparisons); its configuration name must match
# run_gemm.py's apart from an optional suffix, given with --suffix.
# Only Verilator numbers are performance numbers: gvsoc is not cycle-accurate.
set -euo pipefail

SB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SB_ROOT"

driver=run_gemm.py
suffix=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --driver) driver="$2"; shift 2 ;;
    --suffix) suffix="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
if [[ $# -lt 6 ]]; then
  echo "usage: $0 [--driver script --suffix s] N M K n m k [xdsl|library]" >&2
  exit 1
fi
N=$1 M=$2 K=$3 n=$4 m=$5 k=$6 mk=${7:-library}
[[ "$mk" == library ]] && short=lib || short=xdsl

name="gemm_M${M}_N${N}_K${K}_t${m}x${n}x${k}_pipe_${mk}${suffix}"
logs="out/$name/verilator/logs"
rm -f "$logs/trace_hart_00008.dasm" "$logs"/*.json "$logs"/*.csv

# Forward run_gemm.py's opt-in experiment switches into the container.
env_args=()
for var in GEMM_STAGGER_N GEMM_PREFETCH GEMM_HW_BARRIER SNITCH_FULL_TRACE; do
  [[ -n "${!var:-}" ]] && env_args+=(-e "$var=${!var}")
done

if ! out=$(docker compose exec -T "${env_args[@]}" snitch "./$driver" "$N" "$M" "$K" "$n" "$m" "$k" \
             true verilator "$mk" true 2>&1) ||
   ! grep -q "^\[+\] verified" <<<"$out"; then
  echo "t${m}x${n}x${k}${suffix}: FAILED"
  grep -E '^\[!\]|mismatch|error:' <<<"$out" | head -3
  exit 1
fi

exp="${M}-${N}-${K}w${m}-${n}-${k}-${short}${suffix}"
bash snitch_cluster/myrtle-experiments/extract_dma_csv.sh \
  "$PWD/$logs/trace_hart_00008.dasm" "$exp" "$M" "$N" "$K" "$m" "$n" "$k" \
  > "$logs/extract.log" 2>&1 ||
  { echo "t${m}x${n}x${k}${suffix}: verified, extraction failed (see $logs/extract.log)"; exit 1; }

dma=$(python3 -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['dma'])" \
      "$logs/$exp.csv")
echo "t${m}x${n}x${k}${suffix}: verified, dma=$dma"
