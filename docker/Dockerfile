FROM python:3.14-slim-trixie

# USER root

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev

RUN apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /opt/gaia_tsf

COPY requirements.txt /tmp
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# USER postgres
