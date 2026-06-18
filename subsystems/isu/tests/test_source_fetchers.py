"""Unit tests for bulk_upload_scheduler/source_fetchers.py."""

import ftplib
import io
import stat
from unittest.mock import MagicMock, call, patch

import pytest

from subsystems.isu.bulk_upload_scheduler.source_fetchers import (
    fetch_from_ftp,
    fetch_from_https,
    fetch_from_s3,
    fetch_from_sftp,
)


@pytest.fixture
def mock_logger() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# HTTPS
# ---------------------------------------------------------------------------


class TestFetchFromHttps:
    """Tests for fetch_from_https."""

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.requests')
    def test_SFC_001_successful_download(self, mock_requests, mock_logger) -> None:
        """A valid URL is downloaded and returned as (filename, bytes)."""
        mock_response = MagicMock()
        mock_response.content = b'timestamp,value\n2026-01-01,1.0'
        mock_requests.get.return_value = mock_response

        results = fetch_from_https(
            urls=['https://example.org/sensors/latest.csv'], logger=mock_logger
        )

        assert results == [('latest.csv', b'timestamp,value\n2026-01-01,1.0')]
        mock_logger.error.assert_not_called()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.requests')
    def test_SFC_002_http_error_is_logged_and_skipped(
        self, mock_requests, mock_logger
    ) -> None:
        """An HTTP error is logged and the failing URL is skipped."""
        mock_requests.RequestException = Exception
        mock_requests.get.side_effect = Exception('404 Not Found')

        results = fetch_from_https(
            urls=['https://example.org/missing.csv'], logger=mock_logger
        )

        assert results == []
        mock_logger.error.assert_called_once()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.requests')
    def test_SFC_003_partial_failure_returns_successful_urls(
        self, mock_requests, mock_logger
    ) -> None:
        """When one URL fails, the other URLs are still returned."""
        ok_response = MagicMock()
        ok_response.content = b'data'

        def side_effect(url, timeout):
            if 'bad' in url:
                raise Exception('connection error')
            return ok_response

        mock_requests.get.side_effect = side_effect
        mock_requests.RequestException = Exception

        results = fetch_from_https(
            urls=[
                'https://example.org/good.csv',
                'https://example.org/bad.csv',
            ],
            logger=mock_logger,
        )

        assert len(results) == 1
        assert results[0][0] == 'good.csv'
        mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


class _FakeClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError in tests."""


class TestFetchFromS3:
    """Tests for fetch_from_s3."""

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.botocore')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.boto3')
    def test_SFC_004_successful_list_and_download(
        self, mock_boto3, mock_botocore, mock_logger
    ) -> None:
        """Objects under a prefix are listed and downloaded as (filename, bytes)."""
        mock_botocore.exceptions.ClientError = _FakeClientError

        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {'Contents': [{'Key': 'in-situ/piezometer.csv'}]}
        ]
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=MagicMock(return_value=b'ts,value\n'))
        }

        results = fetch_from_s3(
            bucket='gaia-tsf-bulk-uploads',
            prefix='in-situ/',
            logger=mock_logger,
        )

        assert results == [('piezometer.csv', b'ts,value\n')]
        mock_logger.error.assert_not_called()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.boto3', None)
    def test_SFC_005_boto3_missing_logs_error_and_returns_empty(
        self, mock_logger
    ) -> None:
        """When boto3 is not installed, an error is logged and [] is returned."""
        results = fetch_from_s3(bucket='gaia-tsf-bulk-uploads', logger=mock_logger)

        assert results == []
        mock_logger.error.assert_called_once()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.botocore')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.boto3')
    def test_SFC_006_client_error_logs_and_returns_empty(
        self, mock_boto3, mock_botocore, mock_logger
    ) -> None:
        """A botocore ClientError is caught, logged, and [] is returned."""
        mock_botocore.exceptions.ClientError = _FakeClientError

        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = _FakeClientError('Access Denied')

        results = fetch_from_s3(bucket='gaia-tsf-bulk-uploads', logger=mock_logger)

        assert results == []
        mock_logger.error.assert_called_once()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.botocore')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.boto3')
    def test_SFC_007_prefix_directory_keys_are_skipped(
        self, mock_boto3, mock_botocore, mock_logger
    ) -> None:
        """Keys ending with '/' (S3 pseudo-directories) are not downloaded."""
        mock_botocore.exceptions.ClientError = _FakeClientError

        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {'Contents': [{'Key': 'in-situ/'}, {'Key': 'in-situ/sensor.csv'}]}
        ]
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=MagicMock(return_value=b'data'))
        }

        results = fetch_from_s3(bucket='gaia-tsf-bulk-uploads', logger=mock_logger)

        assert len(results) == 1
        assert results[0][0] == 'sensor.csv'


# ---------------------------------------------------------------------------
# FTP
# ---------------------------------------------------------------------------


class TestFetchFromFtp:
    """Tests for fetch_from_ftp."""

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.ftplib.FTP')
    def test_SFC_008_successful_list_and_download(
        self, mock_ftp_class, mock_logger
    ) -> None:
        """Files in the remote directory are listed and downloaded as (filename, bytes)."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        mock_ftp_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.return_value = ['sensor.csv']

        def fake_retrbinary(cmd, callback):
            callback(b'ts,value\n')

        mock_ftp.retrbinary.side_effect = fake_retrbinary

        results = fetch_from_ftp(
            host='ftp.example.org',
            user='gaia',
            password='secret',
            remote_dir='/incoming',
            logger=mock_logger,
        )

        assert results == [('sensor.csv', b'ts,value\n')]
        mock_logger.error.assert_not_called()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.ftplib.FTP')
    def test_SFC_009_directory_entries_are_skipped(
        self, mock_ftp_class, mock_logger
    ) -> None:
        """Entries that raise error_perm on RETR (likely sub-directories) are skipped."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__ = MagicMock(return_value=mock_ftp)
        mock_ftp_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ftp.nlst.return_value = ['subdir', 'sensor.csv']

        def fake_retrbinary(cmd, callback):
            if 'subdir' in cmd:
                raise ftplib.error_perm('550 Is a directory')
            callback(b'data')

        mock_ftp.retrbinary.side_effect = fake_retrbinary

        results = fetch_from_ftp(
            host='ftp.example.org',
            user='gaia',
            password='secret',
            logger=mock_logger,
        )

        assert len(results) == 1
        assert results[0][0] == 'sensor.csv'

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.ftplib.FTP')
    def test_SFC_010_connection_failure_logs_and_returns_empty(
        self, mock_ftp_class, mock_logger
    ) -> None:
        """An FTP connection error is logged and [] is returned."""
        mock_ftp_class.return_value.__enter__.side_effect = ftplib.error_temp(
            'Connection refused'
        )

        results = fetch_from_ftp(
            host='ftp.example.org',
            user='gaia',
            password='secret',
            logger=mock_logger,
        )

        assert results == []
        mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# SFTP
# ---------------------------------------------------------------------------


class TestFetchFromSftp:
    """Tests for fetch_from_sftp."""

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.stat')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.paramiko')
    def test_SFC_011_successful_download_with_password(
        self, mock_paramiko, mock_stat, mock_logger
    ) -> None:
        """Files in the remote directory are downloaded via password auth."""
        mock_transport = MagicMock()
        mock_paramiko.Transport.return_value = mock_transport

        mock_sftp = MagicMock()
        mock_paramiko.SFTPClient.from_transport.return_value = mock_sftp

        entry = MagicMock()
        entry.filename = 'sensor.csv'
        mock_sftp.listdir_attr.return_value = [entry]
        mock_stat.S_ISDIR.return_value = False

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = b'ts,value\n'
        mock_sftp.open.return_value = mock_file

        results = fetch_from_sftp(
            host='sftp.example.org',
            user='gaia',
            password='secret',
            remote_dir='/incoming',
            logger=mock_logger,
        )

        assert results == [('sensor.csv', b'ts,value\n')]
        mock_transport.connect.assert_called_once_with(username='gaia', password='secret')
        mock_logger.error.assert_not_called()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.stat')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.paramiko')
    def test_SFC_012_key_based_auth_is_used_when_key_path_given(
        self, mock_paramiko, mock_stat, mock_logger
    ) -> None:
        """When key_path is provided, RSAKey is loaded and used instead of password."""
        mock_transport = MagicMock()
        mock_paramiko.Transport.return_value = mock_transport

        mock_sftp = MagicMock()
        mock_paramiko.SFTPClient.from_transport.return_value = mock_sftp
        mock_sftp.listdir_attr.return_value = []

        fake_key = MagicMock()
        mock_paramiko.RSAKey.from_private_key_file.return_value = fake_key

        fetch_from_sftp(
            host='sftp.example.org',
            user='gaia',
            key_path='/secrets/id_rsa',
            remote_dir='/incoming',
            logger=mock_logger,
        )

        mock_paramiko.RSAKey.from_private_key_file.assert_called_once_with(
            '/secrets/id_rsa'
        )
        mock_transport.connect.assert_called_once_with(username='gaia', pkey=fake_key)

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.paramiko', None)
    def test_SFC_013_paramiko_missing_logs_error_and_returns_empty(
        self, mock_logger
    ) -> None:
        """When paramiko is not installed, an error is logged and [] is returned."""
        results = fetch_from_sftp(
            host='sftp.example.org', user='gaia', password='secret', logger=mock_logger
        )

        assert results == []
        mock_logger.error.assert_called_once()

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.stat')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.paramiko')
    def test_SFC_014_directory_entries_are_skipped(
        self, mock_paramiko, mock_stat, mock_logger
    ) -> None:
        """Entries for which S_ISDIR returns True are not downloaded."""
        mock_transport = MagicMock()
        mock_paramiko.Transport.return_value = mock_transport

        mock_sftp = MagicMock()
        mock_paramiko.SFTPClient.from_transport.return_value = mock_sftp

        dir_entry = MagicMock()
        dir_entry.filename = 'subdir'
        file_entry = MagicMock()
        file_entry.filename = 'sensor.csv'
        mock_sftp.listdir_attr.return_value = [dir_entry, file_entry]

        mock_stat.S_ISDIR.side_effect = lambda mode: mode == dir_entry.st_mode

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = b'data'
        mock_sftp.open.return_value = mock_file

        results = fetch_from_sftp(
            host='sftp.example.org',
            user='gaia',
            password='secret',
            remote_dir='/incoming',
            logger=mock_logger,
        )

        assert len(results) == 1
        assert results[0][0] == 'sensor.csv'

    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.stat')
    @patch('subsystems.isu.bulk_upload_scheduler.source_fetchers.paramiko')
    def test_SFC_015_ssh_exception_logs_and_returns_empty(
        self, mock_paramiko, mock_stat, mock_logger
    ) -> None:
        """An SSHException is caught, logged, and [] is returned. Transport is closed."""
        mock_paramiko.SSHException = Exception
        mock_transport = MagicMock()
        mock_paramiko.Transport.return_value = mock_transport
        mock_transport.connect.side_effect = Exception('Authentication failed')

        results = fetch_from_sftp(
            host='sftp.example.org',
            user='gaia',
            password='wrong',
            logger=mock_logger,
        )

        assert results == []
        mock_logger.error.assert_called_once()
        mock_transport.close.assert_called_once()
