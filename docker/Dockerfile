FROM ghcr.io/stac-utils/pgstac:v0.9.9

USER root

RUN apt-get update && apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /opt/gaia_tsf

COPY requirements.txt /tmp
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

USER postgres
