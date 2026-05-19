# ---- Builder stage ----
FROM python:3.14-slim-trixie AS builder

RUN apt-get update
RUN apt-get install -y git autoconf libtiff5-dev libhdf5-dev liblapack-dev
RUN apt-get install -y g++ gfortran make
RUN apt-get install -y gmt gmt-dcw gmt-gshhg libgmt-dev

RUN git clone --branch 6.6 https://github.com/gmtsar/gmtsar /tmp/gmtsar
WORKDIR /tmp/gmtsar
RUN autoconf && autoupdate
RUN CPPFLAGS="-D_POSIX_C_SOURCE=200809L" CFLAGS="-O2 -fcommon" ./configure --prefix=/opt/gmtsar
RUN make && make install

# ---- Runtime stage ----
FROM python:3.14-slim-trixie

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev libgmt6t64

RUN apt-get install -y python3-pip
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gaia_tsf

COPY requirements.txt /tmp
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# copy gtmsar binaries
COPY --from=builder /opt/gmtsar /opt/gmtsar
ENV PATH="/opt/gmtsar/bin:${PATH}"
