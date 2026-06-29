#!/usr/bin/env bash
# wait-migrate-start.sh

# 1) Waiting for DB
/usr/local/bin/wait-for-db.sh

# 2) migrate
pypgstac migrate --dsn postgresql://stac:stac@pgstacdb:5432/stac

# 3) create default collections
python3 /usr/local/bin/create_default_collections.py

# 4) start STAC API
echo "Starting STAC FastAPI with Gunicorn..."
exec gunicorn stac_fastapi.pgstac.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 2 \
  --bind 0.0.0.0:8000 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
