import sys
import pytest
from unittest.mock import patch
import json
from pathlib import Path

# Add lambda folder to sys.path so we can import handler
sys.path.append(str(Path(__file__).resolve().parent.parent / 'lambda'))

from proxy.handler import sign_url, validate_user, s3_parse_s3_url


class TestProxyHandler:
    """Test suite for the sign_url Lambda handler."""

    # ---------------------------------
    # Helper method to create mock API Gateway event
    # ---------------------------------
    @staticmethod
    def make_event(
        auth='Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem',
        s3url='s3://gaia-tsf-private/file.txt',
    ):
        """
        Create a mock API Gateway event with headers and query parameters.
        """
        return {
            'headers': {'Authorization': auth},
            'queryStringParameters': {'s3url': s3url},
        }

    # ---------------------------------
    # Test validate_user function
    # ---------------------------------
    def test_validate_user(self):
        """
        Test the validate_user function for correct auth validation.
        """
        # Valid token should return True
        assert validate_user('Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem') is True
        # Invalid token should return False
        assert validate_user('wrong') is False

    # ---------------------------------
    # Test s3_parse_s3_url function
    # ---------------------------------
    def test_s3_parse_s3_url(self):
        """
        Test parsing of S3 URLs into bucket and key.
        """
        # Normal S3 URL with bucket and key
        bucket, key = s3_parse_s3_url('s3://mybucket/path/to/file.txt')
        assert bucket == 'mybucket'
        assert key == 'path/to/file.txt'

        # S3 URL with only bucket
        bucket, key = s3_parse_s3_url('s3://mybucket')
        assert bucket == 'mybucket'
        assert key == ''

        # Invalid URL should raise ValueError
        with pytest.raises(ValueError):
            s3_parse_s3_url('http://mybucket/file.txt')

    # ---------------------------------
    # Test sign_url function success (mocked presigned URL)
    # ---------------------------------
    @patch('proxy.handler.s3.generate_presigned_url')
    def test_sign_url_success(self, mock_presign):
        """
        Test sign_url returns 200 and correct presigned URL when authorized.
        """
        # Mock the boto3 generate_presigned_url method
        mock_presign.return_value = 'https://signed-url'

        event = self.make_event()
        response = sign_url(event, None)

        # Parse JSON body
        body = json.loads(response['body'])

        # Check status and URL
        assert response['statusCode'] == 200
        assert body['url'] == 'https://signed-url'

    # ---------------------------------
    # Test unauthorized access
    # ---------------------------------
    def test_sign_url_unauthorized(self):
        """
        Test sign_url returns 403 when authorization fails.
        """
        event = self.make_event(auth='wrong')
        response = sign_url(event, None)

        # Check status and error in body
        assert response['statusCode'] == 403
        body = json.loads(response['body'])
        assert 'error' in body

    # ---------------------------------
    # Test access to wrong bucket
    # ---------------------------------
    def test_sign_url_wrong_bucket(self):
        """
        Test sign_url returns 400 when bucket is not allowed.
        """
        event = self.make_event(s3url='s3://other-bucket/file.txt')
        response = sign_url(event, None)

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'Requested bucket' in body['error']

    # ---------------------------------
    # Test missing key in S3 URL
    # ---------------------------------
    def test_sign_url_missing_key(self):
        """
        Test sign_url returns 400 when S3 key is missing.
        """
        event = self.make_event(s3url='s3://gaia-tsf-private/')
        response = sign_url(event, None)

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert "Missing 'file'" in body['error']
