# Spatial-Data-Infrastructure
Central repository for spatial and temporal datasets with geospatial services.

## Pure unit tests
```sh
cd system-root
docker run --rm -v `pwd`/subsystems:/opt/gaia_tsf gaia_tsf_eou:latest  python3 -m pytest /opt/gaia_tsf/sdi/tests/test_stac_handler.py -sv
docker run --rm -v `pwd`/subsystems:/opt/gaia_tsf gaia_tsf_eou:latest  python3 -m pytest /opt/gaia_tsf/sdi/tests/test_proxy_handler.py -sv
```
