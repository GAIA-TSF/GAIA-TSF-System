#!/usr/bin/env bash

until python3 -c "
import psycopg
import sys
try:
    conn = psycopg.connect(host='pgstacdb', port=5432, dbname='stac', user='stac', password='stac')
    conn.close()
except Exception:
    sys.exit(1)
"; do
    echo 'Waiting for database...'
    sleep 1
done
