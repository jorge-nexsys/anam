# ─── AnamDB Docker Image ───────────────────────────────────────────────
# Multi-stage build: compile in Rust builder, run in minimal runtime.
#
# Build:  docker build -t anamdb .
# Run:    docker run -p 8080:8080 anamdb
# GPU:    docker run --gpus all -p 8080:8080 anamdb serve --port 0.0.0.0:8080 --gpu

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM rust:latest AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release --bin anam

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM debian:bookworm-slim

LABEL maintainer="Jorge Martinez"
LABEL org.opencontainers.image.source="https://github.com/jorge-nexsys/anam"
LABEL org.opencontainers.image.description="AnamDB — the AI-native neurosymbolic database engine"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/anam /usr/local/bin/anam

# Create default data directory and initialize it.
RUN mkdir -p /data/anamdb && \
    anam init /data/anamdb

WORKDIR /data/anamdb

# Default: start the server on port 8080.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD echo '{"method":"health"}' | nc -q 1 localhost 8080 | grep -q SERVING || exit 1

ENTRYPOINT ["anam"]
CMD ["serve", "--port", "0.0.0.0:8080"]
