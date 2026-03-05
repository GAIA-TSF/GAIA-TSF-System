import psycopg2
import pytest

from config import *

class TestPGSTAC:
    @pytest.fixture(scope='module')
    def db_connection(self):
        conn = psycopg2.connect(**DB_CONFIG_STAC)
        yield conn
        conn.close()

    def test_connection(self, db_connection):
        with db_connection.cursor() as cur:
            cur.execute('SELECT 1;')
            result = cur.fetchone()
            assert result[0] == 1
