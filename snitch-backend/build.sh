#!/usr/bin/env bash
# Builds/verifies the three moving pieces the matmul driver needs:
# LLVM (mlir-opt/mlir-translate/llc), the xdsl-opt venv, and gvsoc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

die() { echo "[!] $*" >&2; exit 1; }
info() { echo "[+] $*"; }

DO_LLVM=0
DO_XDSL=0
DO_GVSOC=0
REBUILD_LLVM=0
RECONFIGURE_LLVM=0
FORCE=0

if [ "$#" -eq 0 ]; then
  DO_LLVM=1; DO_XDSL=1; DO_GVSOC=1
fi

for arg in "$@"; do
  case "$arg" in
    --all) DO_LLVM=1; DO_XDSL=1; DO_GVSOC=1 ;;
    --llvm) DO_LLVM=1 ;;
    --xdsl) DO_XDSL=1 ;;
    --gvsoc) DO_GVSOC=1 ;;
    --rebuild-llvm) DO_LLVM=1; REBUILD_LLVM=1 ;;
    --reconfigure-llvm) DO_LLVM=1; RECONFIGURE_LLVM=1 ;;
    --force) FORCE=1 ;;
    *) die "unknown flag: $arg (expected --llvm|--xdsl|--gvsoc|--all|--rebuild-llvm|--reconfigure-llvm|--force)" ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 1: LLVM (mlir-opt / mlir-translate / llc)
# ---------------------------------------------------------------------------
if [ "$DO_LLVM" -eq 1 ]; then
  if [ "$RECONFIGURE_LLVM" -eq 1 ]; then
    BUILD_CONTAINER="$LLVM_ROOT/build-container"
    info "Reconfiguring a fresh build at $BUILD_CONTAINER (container clang/lld, from scratch)."
    mkdir -p "$BUILD_CONTAINER"
    cmake -S "$LLVM_ROOT/llvm" -B "$BUILD_CONTAINER" -GNinja \
      -DLLVM_TARGETS_TO_BUILD="Native;RISCV" \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release
    ninja -C "$BUILD_CONTAINER" mlir-opt mlir-translate llc
    info "Built at $BUILD_CONTAINER. Re-run with LLVM_BUILD=$BUILD_CONTAINER to use it."
  elif [ "$REBUILD_LLVM" -eq 1 ]; then
    CACHE="$LLVM_BUILD/CMakeCache.txt"
    [ -f "$CACHE" ] || die "no CMakeCache.txt at $LLVM_BUILD -- did you mount llvm-project at the exact same path it was configured at?"
    CACHED_DIR="$(grep '^CMAKE_CACHEFILE_DIR:INTERNAL=' "$CACHE" | cut -d= -f2-)"
    [ "$CACHED_DIR" = "$LLVM_BUILD" ] || die "CMakeCache.txt was configured for '$CACHED_DIR', not '$LLVM_BUILD' -- the bind mount must land at the identical host path it was built at, or use --reconfigure-llvm instead."
    if grep -q 'CMAKE_CXX_COMPILER_ID "IntelLLVM"' "$LLVM_BUILD"/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null; then
      info "Note: this build tree was configured with the host's Intel oneAPI clang. Incremental objects will instead be compiled with this container's clang via the /usr/lib64/ccache shims -- same C++ ABI, expected to link fine, but not the original compiler."
    fi
    info "Rebuilding mlir-opt/mlir-translate/llc at $LLVM_BUILD"
    ninja -C "$LLVM_BUILD" mlir-opt mlir-translate llc
  else
    [ -x "$MLIR_OPT" ] || die "$MLIR_OPT not found/executable -- either the mount is missing, or pass --rebuild-llvm/--reconfigure-llvm"
    [ -x "$MLIR_TRANSLATE" ] || die "$MLIR_TRANSLATE not found/executable"
    [ -x "$LLC" ] || die "$LLC not found/executable"
    "$MLIR_OPT" --version >/dev/null || die "mlir-opt --version failed"
    info "LLVM tools present and runnable at $LLVM_BUILD/bin (use --rebuild-llvm to incrementally rebuild, --reconfigure-llvm to build fresh)."
  fi
fi

ensure_gvsoc_submodules() {
  if [ ! -f "$GVSOC_ROOT/gapy/bin/gapy" ]; then
    info "Initializing gvsoc submodules"
    git -C "$LLVM_ROOT" submodule update --init "snitch-backend/gvsoc"
    git -C "$GVSOC_ROOT" submodule update --init --recursive
  fi
}

ensure_python_env() {
  if [ "$FORCE" -eq 1 ] && [ -d "$VENV" ]; then
    rm -rf "$VENV"
  fi
  if [ ! -x "$VENV/bin/xdsl-opt" ]; then
    ensure_gvsoc_submodules
    info "Creating venv at $VENV and installing xdsl-opt + gvsoc's Python deps"
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" -e "$SB_ROOT/xdsl"
    [ -f "$GVSOC_ROOT/requirements.txt" ] && uv pip install --python "$VENV/bin/python" -r "$GVSOC_ROOT/requirements.txt"
    [ -f "$GVSOC_ROOT/gapy/requirements.txt" ] && uv pip install --python "$VENV/bin/python" -r "$GVSOC_ROOT/gapy/requirements.txt"
    [ -f "$GVSOC_ROOT/core/requirements.txt" ] && uv pip install --python "$VENV/bin/python" -r "$GVSOC_ROOT/core/requirements.txt"
  else
    info "venv already has xdsl-opt at $VENV (use --force to reinstall)"
  fi
}

# ---------------------------------------------------------------------------
# Step 2: Python venv + xdsl-opt
# ---------------------------------------------------------------------------
if [ "$DO_XDSL" -eq 1 ]; then
  ensure_python_env

  # --help word-wraps the pass list (sometimes mid-identifier, at a hyphen),
  # so strip whitespace/newlines before matching.
  "$XDSL_OPT" --help | tr -d ' \n' | grep -q 'test-lower-snitch-stream-to-asm' || \
    die "xdsl-opt at $XDSL_OPT does not provide the passes ConvertToRISCV.cpp expects (test-lower-snitch-stream-to-asm missing) -- check the xdsl submodule pin (should be 024068cdbfe3c35be2ce7d21ef7b86c1b028a7b1)."
  info "xdsl-opt pipeline compatibility check passed."
fi

# ---------------------------------------------------------------------------
# Step 3: gvsoc
# ---------------------------------------------------------------------------
if [ "$DO_GVSOC" -eq 1 ]; then
  if [ "$FORCE" -eq 1 ]; then
    rm -rf "$GVSOC_WORKDIR"
  fi
  if [ -z "$(find "$GVSOC_WORKDIR/install" -iname '*.so*' -print -quit 2>/dev/null)" ]; then
    ensure_gvsoc_submodules
    ensure_python_env   # `make` shells out to gapy, which needs the venv's deps (e.g. six) on PATH
    info "Building gvsoc (TARGETS=snitch) into $GVSOC_WORKDIR"
    GVSOC_WORKDIR="$GVSOC_WORKDIR" make -C "$GVSOC_ROOT" all TARGETS=snitch "CMAKE_FLAGS=-j$(nproc)"
  else
    info "gvsoc already built at $GVSOC_WORKDIR/install (use --force to rebuild)"
  fi

  export PATH="$GVSOC_WORKDIR/install/bin:$PATH"
  gvrun --target snitch --work-dir "$OUT/.gvsoc_smoke_test" commands >/dev/null || \
    die "gvrun --target snitch commands failed -- gvsoc build is incomplete or broken"
  info "gvrun smoke test passed (snitch target resolved)."
fi

info "build.sh done."
