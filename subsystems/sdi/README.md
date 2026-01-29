# Spatial-Data-Infrastructure
Central repository for spatial and temporal datasets with geospatial services.

# Unit tests

## Pure unit tests
```sh
cd system-root
docker run --rm -v `pwd`/subsystems:/opt/gaia_tsf gaia_tsf_eou:latest  python3 -m pytest /opt/gaia_tsf/sdi/tests/test_stac_handler.py -sv
docker run --rm -v `pwd`/subsystems:/opt/gaia_tsf gaia_tsf_eou:latest  python3 -m pytest /opt/gaia_tsf/sdi/tests/test_proxy_handler.py -sv
```

## Using LocalStack docker simulating AWS infrastructure

```sh
cd system-root/docker/
docker-compose run --rm app python3 -m pytest /opt/gaia_tsf/sdi/tests/test_sign_url.py -sv
```
