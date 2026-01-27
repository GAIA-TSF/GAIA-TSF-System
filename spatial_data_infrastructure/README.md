# Spatial-Data-Infrastructure
Central repository for spatial and temporal datasets with geospatial services.

# Unit tests

cd system-root/docker/
docker-compose run --rm app python3 -m pytest /opt/gaia_tsf/spatial_data_infrastructure/tests/test_sign_url.py -sv
