
# We take advatnage of the RISC-V build already present in the Quidditch image so we dont have to build again
FROM ghcr.io/opencompl/quidditch/toolchain:main AS toolchain

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000
ARG LLVM_PROJECT_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang lld cmake ninja-build ccache \
    git git-lfs curl ca-certificates \
    python3 python3-venv \
    zlib1g-dev libzstd1 pkg-config file less \
    && rm -rf /var/lib/apt/lists/*

# Copying from the Quidditch toolchain image to avoid rebuilding the RISC-V toolchain
COPY --from=toolchain /opt/quidditch-toolchain /opt/quidditch-toolchain
ENV QUIDDITCH_TOOLCHAIN=/opt/quidditch-toolchain

# Same as the Quidditch image, but we don't need the full uv image, just the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir -p /usr/lib64/ccache && \
    ln -s /usr/bin/ccache /usr/lib64/ccache/clang && \
    ln -s /usr/bin/ccache /usr/lib64/ccache/clang++

RUN groupadd -g "$GID" dev && \
    useradd -u "$UID" -g "$GID" -m -s /bin/bash dev

ENV CCACHE_DIR=/home/dev/.ccache
RUN mkdir -p "$CCACHE_DIR" && chown "$UID:$GID" "$CCACHE_DIR"

USER dev
RUN git config --global --add safe.directory '*'

WORKDIR ${LLVM_PROJECT_DIR}/snitch-backend
