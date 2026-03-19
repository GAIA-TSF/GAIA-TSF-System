import psycopg
import pytest

from lib.config import SettingsReader


class TestPGSTAC:
    @pytest.fixture(scope='module')
    def db_connection(self):
        settings = SettingsReader()

        conn = psycopg.connect(**settings['sdi']['stac']['db'])
        yield conn
        conn.close()

    def test_connection(self, db_connection):
        with db_connection.cursor() as cur:
            cur.execute('SELECT 1;')
            result = cur.fetchone()
            assert result[0] == 1
