import ftplib
import io
import stat
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

import requests

# --- Optional Dependencies ---
try:
    import boto3
    import botocore.exceptions
except ImportError:
    boto3 = None
    botocore = None

try:
    import paramiko
except ImportError:
    paramiko = None


def fetch_from_https(urls: List[str], logger: Any = None) -> List[Tuple[str, bytes]]:
    """
    Download a fixed list of HTTPS URLs and return their filenames and raw bytes.

    :param urls: List of file URLs to download.
    :type urls: List[str]
    :param logger: Optional injected logger.
    :type logger: Any
    :return: List of (filename, content) tuples for successfully downloaded files.
    :rtype: List[Tuple[str, bytes]]
    """
    results = []
    for url in urls:
        try:
            if logger:
                logger.debug(f'Fetching {url} over HTTPS...')
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            filename = urlparse(url).path.rsplit('/', 1)[-1] or url
            results.append((filename, response.content))
            if logger:
                logger.info(
                    f'Downloaded {filename} ({len(response.content)} bytes) via HTTPS.'
                )
        except requests.RequestException as e:
            if logger:
                logger.error(f'Failed to fetch {url} over HTTPS: {e}')
    return results


def fetch_from_s3(
    bucket: str,
    prefix: str = '',
    region_name: Optional[str] = None,
    logger: Any = None,
) -> List[Tuple[str, bytes]]:
    """
    List and download all objects found under a prefix in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :type bucket: str
    :param prefix: Key prefix to filter objects (acts like a remote directory).
    :type prefix: str
    :param region_name: Optional AWS region override.
    :type region_name: Optional[str]
    :param logger: Optional injected logger.
    :type logger: Any
    :return: List of (filename, content) tuples for successfully downloaded objects.
    :rtype: List[Tuple[str, bytes]]
    """
    if boto3 is None:
        if logger:
            logger.error('boto3 is not installed. Cannot fetch from S3.')
        return []

    results = []
    try:
        if logger:
            logger.debug(f'Connecting to S3 bucket {bucket!r} (prefix={prefix!r})...')
        s3 = boto3.client('s3', region_name=region_name)
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                if logger:
                    logger.debug(f'Downloading s3://{bucket}/{key}...')
                response = s3.get_object(Bucket=bucket, Key=key)
                filename = key.rsplit('/', 1)[-1]
                content = response['Body'].read()
                results.append((filename, content))
                if logger:
                    logger.info(
                        f'Downloaded {filename} ({len(content)} bytes) from S3.'
                    )
    except botocore.exceptions.ClientError as e:
        if logger:
            logger.error(f'Failed to fetch objects from S3 bucket {bucket}: {e}')
    return results


def fetch_from_ftp(
    host: str,
    user: str,
    password: str,
    remote_dir: str = '/',
    port: int = 21,
    logger: Any = None,
) -> List[Tuple[str, bytes]]:
    """
    Connect to an FTP server and download every file found in a remote directory.

    :param host: FTP server hostname or IP address.
    :type host: str
    :param user: FTP username.
    :type user: str
    :param password: FTP password.
    :type password: str
    :param remote_dir: Remote directory to scan for files.
    :type remote_dir: str
    :param port: FTP server port.
    :type port: int
    :param logger: Optional injected logger.
    :type logger: Any
    :return: List of (filename, content) tuples for successfully downloaded files.
    :rtype: List[Tuple[str, bytes]]
    """
    results = []
    try:
        if logger:
            logger.debug(f'Connecting to FTP {host}:{port}{remote_dir}...')
        with ftplib.FTP() as ftp:
            ftp.connect(host=host, port=port, timeout=30)
            ftp.login(user=user, passwd=password)
            ftp.cwd(remote_dir)

            for filename in ftp.nlst():
                buffer = io.BytesIO()
                try:
                    if logger:
                        logger.debug(f'Downloading {filename} via FTP...')
                    ftp.retrbinary(f'RETR {filename}', buffer.write)
                except ftplib.error_perm:
                    # Most likely a sub-directory or an inaccessible entry, skip it.
                    continue
                content = buffer.getvalue()
                results.append((filename, content))
                if logger:
                    logger.info(
                        f'Downloaded {filename} ({len(content)} bytes) via FTP.'
                    )
    except ftplib.all_errors as e:
        if logger:
            logger.error(f'Failed to fetch from FTP {host}:{remote_dir}: {e}')
    return results


def fetch_from_sftp(
    host: str,
    user: str,
    password: Optional[str] = None,
    key_path: Optional[str] = None,
    remote_dir: str = '.',
    port: int = 22,
    logger: Any = None,
) -> List[Tuple[str, bytes]]:
    """
    Connect to an SFTP server and download every file found in a remote directory.

    Authenticates with a private key (when ``key_path`` is provided) or a password.

    :param host: SFTP server hostname or IP address.
    :type host: str
    :param user: SFTP username.
    :type user: str
    :param password: SFTP password, used when no private key is provided.
    :type password: Optional[str]
    :param key_path: Path to a private key file for key-based authentication.
    :type key_path: Optional[str]
    :param remote_dir: Remote directory to scan for files.
    :type remote_dir: str
    :param port: SFTP server port.
    :type port: int
    :param logger: Optional injected logger.
    :type logger: Any
    :return: List of (filename, content) tuples for successfully downloaded files.
    :rtype: List[Tuple[str, bytes]]
    """
    if paramiko is None:
        if logger:
            logger.error('paramiko is not installed. Cannot fetch from SFTP.')
        return []

    results = []
    transport = None
    try:
        if logger:
            logger.debug(f'Connecting to SFTP {host}:{port}{remote_dir}...')
        transport = paramiko.Transport((host, port))
        if key_path:
            transport.connect(
                username=user, pkey=paramiko.RSAKey.from_private_key_file(key_path)
            )
        else:
            transport.connect(username=user, password=password)

        sftp = paramiko.SFTPClient.from_transport(transport)
        for entry in sftp.listdir_attr(remote_dir):
            if stat.S_ISDIR(entry.st_mode):
                continue
            remote_path = f'{remote_dir.rstrip("/")}/{entry.filename}'
            if logger:
                logger.debug(f'Downloading {entry.filename} via SFTP...')
            with sftp.open(remote_path, 'rb') as remote_file:
                content = remote_file.read()
            results.append((entry.filename, content))
            if logger:
                logger.info(
                    f'Downloaded {entry.filename} ({len(content)} bytes) via SFTP.'
                )
    except (paramiko.SSHException, OSError) as e:
        if logger:
            logger.error(f'Failed to fetch from SFTP {host}:{remote_dir}: {e}')
    finally:
        if transport:
            transport.close()
    return results
