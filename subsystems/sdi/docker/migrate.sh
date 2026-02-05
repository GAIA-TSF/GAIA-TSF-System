# Manually in the container
docker container ls | grep docker2_db
docker exec -it b034c28732f5 /bin/bash
pypgstac migrate --dsn postgresql://stac:stac@localhost:5432/stac

# Or via docker-compose
docker-compose run --rm stacapi pypgstac migrate --dsn postgresql://stac:stac@db:5432/stac
