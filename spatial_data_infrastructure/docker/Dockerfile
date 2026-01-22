FROM ghcr.io/stac-utils/pgstac:latest

USER root

RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir pypgstac psycopg psycopg_pool && \
    rm -rf /var/lib/apt/lists/*

USER postgres
