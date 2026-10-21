# syntax=docker/dockerfile:1.7
# The gliner service (:5800). Build context is the repository root: `make image`.
# No model inside and nothing downloaded at run time: mount an artifact dir (`make export`) read-only at /models.

##### BASES #####
ARG CHEF=lukemathwalker/cargo-chef:latest-rust-1.94-slim-trixie@sha256:c099c03e8a585e114631935ceb8bc7dbbfcfeed19e336feb305c9ef389030f72
ARG RUNTIME=gcr.io/distroless/cc-debian13:nonroot@sha256:b7e81cb1af38d9345680e68bcf564d388f96304e25222030fc8c051faf8565b3

##### TOOLCHAIN #####
# g++ brings libstdc++ for the statically linked ONNX Runtime.
FROM ${CHEF} AS chef
WORKDIR /src
ENV CARGO_TARGET_DIR=/src/target
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

##### RECIPE #####
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

##### BUILD #####
# ONNX Runtime is fetched prebuilt by `ort` here, at build time, and linked statically.
FROM chef AS builder
COPY --from=planner /src/recipe.json recipe.json
RUN cargo chef cook --release --locked --recipe-path recipe.json
COPY . .
RUN cargo build --release --locked --bin gliner \
 && install -D target/release/gliner /out/gliner

##### RUNTIME #####
# Distroless: libc, libstdc++ and certificates only; no shell, no package manager; uid 65532.
FROM ${RUNTIME}
COPY --from=builder /out/gliner /usr/local/bin/gliner
ENV GLINER_MODEL__DIR=/models \
    GLINER_SERVER__LOG__FORMAT=json
USER nonroot
EXPOSE 5800
ENTRYPOINT ["/usr/local/bin/gliner"]
CMD ["serve"]
