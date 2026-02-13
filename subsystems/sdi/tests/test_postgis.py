import os
import psycopg2
import pytest


DB_CONFIG = {
    "host": "postgis",
    "port": 5432,
    "dbname": "geodata",
    "user": "postgres",
    "password": "fevcfQBu3b3CfxFU",
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


def test_schema_exists(db_connection):
    with db_connection.cursor() as cur:
        cur.execute("""
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name = 'test';
                    """)
        result = cur.fetchone()
        assert result is not None, "Schema 'test' does not exist"


def test_table_exists(db_connection):
    with db_connection.cursor() as cur:
        cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'test'
                      AND table_name = 'fakesite_phmeasurements_2025_cybele';
                    """)
        result = cur.fetchone()
        assert result is not None, "Table does not exist"


def test_can_read_data(db_connection):
    with db_connection.cursor() as cur:
        cur.execute("""
                    SELECT COUNT(*)
                    FROM test.fakesite_phmeasurements_2025_cybele;
                    """)
        count = cur.fetchone()[0]
        assert count >= 0
