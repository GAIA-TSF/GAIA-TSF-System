import os
import boto3
from botocore.exceptions import ClientError
import json

from qcl.logger import Logger

LOCALSTACK = os.getenv('LOCALSTACK', '0') == '1'

if LOCALSTACK:
    s3 = boto3.client(
        's3',
        endpoint_url='http://localhost:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )
else:
    s3 = boto3.client('s3')

id = 'SDI'
logger = Logger(subsystem=id)


def validate_user(auth_header):
    """Validates authentification of the user"""
    # TODO base on some access management (Cognito, OAuth ?)
    return auth_header == 'Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem'


def user_has_access(auth_header, key):
    """Validates authorization of the user"""
    # TODO base on some user rights management (PostgreSQL, DynamoDB ?)
    return True


def s3_parse_s3_url(s3_url):
    """Returns bucket name and path from full s3 path"""
    if not s3_url.startswith('s3://'):
        raise ValueError('URL must start with s3://')

    # Remove s3
    path = s3_url[5:]

    # bucket first part, key the rest
    parts = path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''

    return bucket, key


def log_error(err, event):
    """Prints event content so it is logged to Cloud Watch"""
    logger.debug(f'Error on S3 Proxy requested with event {event}')
    logger.debug(f'Error: {err}')


def sign_url(event, context):
    """Returns signer url to allow download file from s3 path"""

    # Check user
    auth_header = event['headers'].get('Authorization')
    if not auth_header or not validate_user(auth_header):
        return {'statusCode': 403, 'body': json.dumps({'error': 'Unauthorized'})}

    # Params
    params = event.get('queryStringParameters', {}) or {}
    s3url = params.get('s3url')
    bucket = 'gaia-tsf-private'

    try:
        bucket_user, key = s3_parse_s3_url(s3url)
    except ClientError as e:
        log_error(e, event)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

    if bucket_user != bucket:
        e = 'Requested bucket is not available'
        log_error(e, event)
        return {'statusCode': 400, 'body': json.dumps({'error': e})}

    if not key:
        e = "Missing 'file' parameter"
        log_error(e, event)
        return {'statusCode': 400, 'body': json.dumps({'error': e})}

    # TODO Access - not functional yet
    if not user_has_access(auth_header, key):
        e = 'Access denied'
        log_error(e, event)
        return {'statusCode': 403, 'body': json.dumps({'error': e})}

    # Signed URL
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600,  # Validity in seconds
        )

    except ClientError as e:
        log_error(e, event)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

    return {'statusCode': 200, 'body': json.dumps({'url': url})}
