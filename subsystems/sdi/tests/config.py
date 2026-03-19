DB_CONFIG_STAC = {
    'host': 'pgstacdb',
    'port': 5432,
    'dbname': 'stac',
    'user': 'stac',
    'password': 'stac',
}


DB_CONFIG_PG = {
    'host': 'postgis',
    'port': 5432,
    'dbname': 'geodata',
    'user': 'postgres',
    'password': 'fevcfQBu3b3CfxFU',
}

STAC_URL = 'http://stacapi:8000'  # Service in docker compose + port in container

INSITU_COLLECTION = 'prague'
INSITU_ITEM_ID = 'measurement_ph_202602'

EO_COLLECTION = 'sample-sentinel2-data'
EO_ITEM_ID = 'sentinel2_prague_20260210'
