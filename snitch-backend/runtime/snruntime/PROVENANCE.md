# snRuntime — vendored artifacts

Everything under this directory (`lib/`, `include/`, `ld/`) is copied, not
built, from the Quidditch repo on the host that originally built it. This lets `snitch-backend/` build/run the matmul
prototype without pulling in Quidditch's `runtime/` CMake project, which otherwise drags in a full IREE `add_subdirectory()` just to configure (`runtime/iree-configuration/CMakeLists.txt` unconditionally does `add_subdirectory(${IREE_SOURCE_DIR} iree EXCLUDE_FROM_ALL)`). We could still build it ourselves in a future version.

## Source

- Upstream: `pulp-platform/snitch_cluster` @ `2fb4c707d3d81c9a6e3008e68b9c2fee716a2ff3`
  (pinned in `quidditch/.gitmodules`).
