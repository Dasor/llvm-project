
# We take advatnage of the RISC-V build already present in the Quidditch image so we dont have to build again
FROM ghcr.io/opencompl/quidditch/toolchain:main AS toolchain

# Same as the snitch-cluster image
FROM hpretl/iic-osic-tools:2026.02 AS oseda

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000
ARG LLVM_PROJECT_DIR
ARG BENDER_VERSION=0.31.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang lld cmake ninja-build ccache \
    git git-lfs curl wget ca-certificates \
    python3 python3-venv \
    zlib1g-dev libzstd1 pkg-config file less \
    patch device-tree-compiler mold binutils \
    && rm -rf /var/lib/apt/lists/*

# Copying from the Quidditch toolchain image to avoid rebuilding the RISC-V toolchain
COPY --from=toolchain /opt/quidditch-toolchain /opt/quidditch-toolchain
ENV QUIDDITCH_TOOLCHAIN=/opt/quidditch-toolchain

# Verilator, copied from the oseda image (see the FROM above) instead of installed
# from a PPA -- this is exactly what snitch_cluster's own container build does.
COPY --from=oseda /foss/tools/verilator /opt/verilator
ENV PATH=/opt/verilator/bin:${PATH}

# bender
RUN wget -q "https://github.com/pulp-platform/bender/releases/download/v${BENDER_VERSION}/bender-${BENDER_VERSION}-x86_64-linux-gnu-ubuntu24.04.tar.gz" \
    -O /tmp/bender.tar.gz && \
    tar xzf /tmp/bender.tar.gz -C /usr/local/bin && \
    rm /tmp/bender.tar.gz

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
RUN git config --global --add safe.directory '*' && \
    git config --global url."https://github.com/".insteadOf "git@github.com:"

WORKDIR ${LLVM_PROJECT_DIR}/snitch-backend
