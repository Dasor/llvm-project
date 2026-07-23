"""Auto-detects this host's cache hierarchy and SIMD vector width from
Linux sysfs/procfs, so GEMM feature engineering
(mlir_tuner/backend.py's GemmKnobFeatureExtractor) can reason about tile
footprints relative to this machine's actual cache sizes -- with zero CLI
flags, since ScheduleEvaluator always ahead-of-time compiles and runs
candidates natively on this same host (see backend.py's ScheduleEvaluator
docstring): "the target" is always just wherever this script runs.

Degrades to fixed conservative fallback values on any failure (missing
sysfs -- non-Linux hosts, some containers; unexpected formats; permission
errors), with a one-time stderr warning, never a crash.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Tuple

_CACHE_ROOT = Path("/sys/devices/system/cpu/cpu0/cache")
_CPUINFO = Path("/proc/cpuinfo")

# Conservative fallback: a modest 2020s x86_64 core, AVX2, used only when
# sysfs/cpuinfo can't be read/parsed at all.
_FALLBACK_L1_BYTES = 32 * 1024
_FALLBACK_L1_WAYS = 8
_FALLBACK_L2_BYTES = 256 * 1024
_FALLBACK_L2_WAYS = 8
_FALLBACK_CACHE_LINE_BYTES = 64
_FALLBACK_VECTOR_WIDTH_BYTES = 32  # AVX2

_SIZE_RE = re.compile(r"^(\d+)([KMG])$")
_VECTOR_FLAGS_TO_BYTES = [
    ("avx512f", 64),
    ("avx2", 32),
    ("avx", 32),
    ("sse2", 16),
]  # checked in this order; first match wins, so wider ISAs take priority


class HostHwInfo(NamedTuple):
    l1_bytes: int
    l1_ways: int
    l2_bytes: int
    l2_ways: int
    cache_line_bytes: int
    vector_width_bytes: int


_FALLBACK = HostHwInfo(
    l1_bytes=_FALLBACK_L1_BYTES,
    l1_ways=_FALLBACK_L1_WAYS,
    l2_bytes=_FALLBACK_L2_BYTES,
    l2_ways=_FALLBACK_L2_WAYS,
    cache_line_bytes=_FALLBACK_CACHE_LINE_BYTES,
    vector_width_bytes=_FALLBACK_VECTOR_WIDTH_BYTES,
)


def _parse_size(text: str) -> int:
    """'32K' -> 32768, '256K' -> 262144, '20M' -> 20971520."""
    m = _SIZE_RE.match(text.strip())
    if not m:
        raise ValueError(f"unrecognized cache size format: {text!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]


def _detect_cache_levels() -> Tuple[Dict[int, Tuple[int, int]], int]:
    """Reads every cpu0/cache/index*/{level,type,size,ways_of_associativity,
    coherency_line_size}, keeping only Data/Unified entries (skipping pure
    Instruction caches). Returns ({level: (size_bytes, ways)}, line_size).
    Raises OSError/ValueError on anything missing/malformed -- caller
    catches and falls back."""
    levels: Dict[int, Tuple[int, int]] = {}
    line_size = None
    index_dirs = sorted(_CACHE_ROOT.glob("index*"))
    if not index_dirs:
        raise FileNotFoundError(f"no {_CACHE_ROOT}/index* entries")
    for index_dir in index_dirs:
        cache_type = (index_dir / "type").read_text().strip()
        if cache_type not in ("Data", "Unified"):
            continue
        level = int((index_dir / "level").read_text().strip())
        size_bytes = _parse_size((index_dir / "size").read_text().strip())
        ways = int((index_dir / "ways_of_associativity").read_text().strip())
        levels[level] = (size_bytes, ways)
        if line_size is None:
            line_size = int((index_dir / "coherency_line_size").read_text().strip())
    if 1 not in levels or 2 not in levels or line_size is None:
        raise ValueError(f"missing L1/L2 Data or Unified cache entries under {_CACHE_ROOT}")
    return levels, line_size


def _detect_vector_width_bytes() -> int:
    """Reads /proc/cpuinfo's first 'flags' line; returns bytes per vector
    register for the widest of avx512f/avx2/avx/sse2 present, else 8
    (scalar)."""
    m = re.search(r"^flags\s*:\s*(.*)$", _CPUINFO.read_text(), re.MULTILINE)
    if not m:
        raise ValueError(f"no 'flags' line found in {_CPUINFO}")
    flags = set(m.group(1).split())
    for flag, width in _VECTOR_FLAGS_TO_BYTES:
        if flag in flags:
            return width
    return 8


@functools.lru_cache(maxsize=1)
def detect() -> HostHwInfo:
    """This host's L1/L2 size+associativity, cache line size, and SIMD
    vector width -- memoized for the process lifetime (this is a
    short-lived CLI tool, not a server, so querying sysfs/cpuinfo once and
    reusing it for every extract() call is exactly the semantics wanted).
    Falls back to fixed conservative values, with a one-time stderr
    warning, if sysfs/cpuinfo are unavailable or unparseable."""
    try:
        levels, line_size = _detect_cache_levels()
        vector_width = _detect_vector_width_bytes()
        l1_bytes, l1_ways = levels[1]
        l2_bytes, l2_ways = levels[2]
        return HostHwInfo(
            l1_bytes=l1_bytes,
            l1_ways=l1_ways,
            l2_bytes=l2_bytes,
            l2_ways=l2_ways,
            cache_line_bytes=line_size,
            vector_width_bytes=vector_width,
        )
    except (OSError, ValueError) as e:
        print(
            f"warning: hw_detect.detect() failed ({e!r}); falling back to "
            f"fixed hardware assumptions: {_FALLBACK}",
            file=sys.stderr,
        )
        return _FALLBACK


def max_square_tile_dim(elem_bytes: int) -> int:
    """A loose upper bound on a single GEMM tile dimension t (assuming a
    roughly cubic tile_m=tile_n=tile_k=t), sized so the combined A+B+C
    tile footprint -- 3 * t^2 * elem_bytes, the same quantity
    backend.py's GemmKnobFeatureExtractor computes as total_tile_bytes --
    fits within this host's actual L2 capacity.

    Deliberately a bound, not an estimate of the optimal tile size: it
    sizes against the whole of L2 with no safety margin held back for
    other tenants (L1 residency, register blocking, OS/other-process
    pressure), and treats the tile as cubic even though the search may
    settle on a skewed m/n/k split whose footprint differs from a cube's
    at the same t. Both push the bound above what a real optimal tile
    would use. That's fine here: this only exists to keep a search
    algorithm (mlir_tuner/run_gbdt_tuner.py) from wasting attempts on
    tiles that can never be cache-resident, not to pick the tile itself --
    the GA/GBDT loop has plenty of headroom below this number to find the
    actual optimum.
    """
    hw = detect()
    return max(int((hw.l2_bytes / (3 * elem_bytes)) ** 0.5), 1)
