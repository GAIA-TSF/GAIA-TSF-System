FROM python:3.14

WORKDIR /opt/gaia_tsf

RUN apt-get update && apt-get -y install libgdal-dev

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt
