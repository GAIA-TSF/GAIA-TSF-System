#!/usr/bin/env bash
# wait-migrate-start.sh

# 1) Waiting for DB
/usr/local/bin/wait-for-db.sh

# 2) migrate
pypgstac migrate --dsn postgresql://stac:stac@pgstacdb:5432/stac

# 3) start STAC API
/opt/venv/bin/stac-fastapi-pgstac serve --host 0.0.0.0
