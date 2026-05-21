# ---- Builder stage ----
FROM python:3.14-slim-trixie AS builder

RUN apt-get update
RUN apt-get install -y git autoconf g++ make gfortran
RUN apt-get install -y libgmt-dev libtiff5-dev

RUN git clone --branch 6.6 https://github.com/gmtsar/gmtsar /tmp/gmtsar
WORKDIR /tmp/gmtsar
RUN autoconf && autoupdate
RUN CPPFLAGS="-D_POSIX_C_SOURCE=200809L" CFLAGS="-O2 -fcommon" ./configure --prefix=/opt/gmtsar
RUN make && make install

# ---- Runtime stage ----
FROM python:3.14-slim-trixie

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev
RUN apt-get install -y gmt libxcb1 libgl1 libglib2.0-0t64
RUN apt-get install -y python3-pip
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gaia_tsf

COPY requirements.txt /tmp
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# copy gtmsar binaries
COPY --from=builder /opt/gmtsar /opt/gmtsar
ENV PATH="/opt/gmtsar/bin:${PATH}"
