import os
import psycopg2
import pytest


DB_CONFIG = {
    "host": "pgstacdb",
    "port": 5432,
    "dbname": "stac",
    "user": "stac",
    "password": "stac",
}


@pytest.fixture(scope="module")
def db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    yield conn
    conn.close()


def test_connection(db_connection):
    with db_connection.cursor() as cur:
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        assert result[0] == 1
