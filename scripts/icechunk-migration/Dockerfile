# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && uv tool install --no-cache \
       "geotessera[migration] @ git+https://github.com/mtelvers/geotessera.git@3a31516ef6fa1e8f208577129db4bb5940ac645d"

ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    ICECHUNK_SOURCE="s3://tessera-embeddings/v1.1/dclimate.icechunk" \
    ZARR_DESTINATION="s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1.1-dclimate" \
    MIGRATION_STATE="s3://us-west-2.opendata.source.coop/tessera/tessera/migrations/v1.1-dclimate" \
    AWS_REGION="us-west-2" \
    UTM_ZONE="" \
    YEAR="" \
    WORKERS="8" \
    IO_CONCURRENCY="16" \
    READ_ROWS="256" \
    SPILL_DIR="/scratch"

RUN mkdir -p /scratch

STOPSIGNAL SIGTERM
ENTRYPOINT exec geotessera-registry zarr-transcode \
    --state "$MIGRATION_STATE" \
    --zones "$UTM_ZONE" \
    --years "$YEAR" \
    --workers "$WORKERS" \
    --io-concurrency "$IO_CONCURRENCY" \
    --read-rows "$READ_ROWS" \
    --spill-dir "$SPILL_DIR" \
    --source-anon \
    --source-region "$AWS_REGION" \
    --store-region "$AWS_REGION" \
    --store-acl bucket-owner-full-control \
    --state-region "$AWS_REGION" \
    --state-acl bucket-owner-full-control
