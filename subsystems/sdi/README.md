# Spatial-Data-Infrastructure
Central repository for spatial and temporal datasets with geospatial services.

# Unit tests

docker run --rm -v `pwd`:/opt/gaia_tsf gaia_tsf_eou:latest  python3 -m pytest /opt/gaia_tsf/spatial_data_infrastructure/tests/test_stac_handler.py -sv
docker run --rm -v `pwd`:/opt/gaia_tsf gaia_tsf_eou:latest  python3 -m pytest /opt/gaia_tsf/spatial_data_infrastructure/tests/test_proxy_handler.py -sv
