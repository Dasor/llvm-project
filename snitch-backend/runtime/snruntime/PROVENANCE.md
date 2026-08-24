# snRuntime — vendored artifacts

Everything under this directory (`lib/`, `include/`, `ld/`) is copied, not
built, from the Quidditch repo on the host that originally built it. This lets `snitch-backend/` build/run the matmul
prototype without pulling in Quidditch's `runtime/` CMake project, which otherwise drags in a full IREE `add_subdirectory()` just to configure (`runtime/iree-configuration/CMakeLists.txt` unconditionally does `add_subdirectory(${IREE_SOURCE_DIR} iree EXCLUDE_FROM_ALL)`). We could still build it ourselves in a future version.

## Source

- Upstream: `pulp-platform/snitch_cluster` @ `2fb4c707d3d81c9a6e3008e68b9c2fee716a2ff3`
  (pinned in `quidditch/.gitmodules`).

Warning: This version has a bug in `sw/runtime/src/start.c` in the function `snrt_init_bss` I manually replaced that piece of code with the commit `71d1c97961ab32677cb80f35bd9a13c94a6d9658` (latest when the fix was applied).