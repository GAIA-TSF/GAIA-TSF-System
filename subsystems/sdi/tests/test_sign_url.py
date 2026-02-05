import os
import sys
import json
import boto3
from pathlib import Path

# Add lambda folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "lambda"))

from proxy.handler import sign_url

# Set environment for LocalStack
os.environ["LOCALSTACK"] = "1"

class TestSignURL:
    @classmethod
    def setup_class(cls):
        """Set up shared resources for all tests in this class"""
        cls.s3 = boto3.client(
            "s3",
            endpoint_url="http://localstack:4566",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
        )
        # Create test bucket, ignore if it already exists
        try:
            cls.s3.create_bucket(Bucket="gaia-tsf-private")
        except cls.s3.exceptions.BucketAlreadyOwnedByYou:
            pass

        # Upload a test object
        cls.s3.put_object(Bucket="gaia-tsf-private", Key="test.txt", Body=b"Hello LocalStack!")

    def test_sign_url_success(self):
        """Test valid signed URL generation"""
        event = {
            "headers": {"Authorization": "Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem"},
            "queryStringParameters": {"s3url": "s3://gaia-tsf-private/test.txt"}
        }
        response = sign_url(event, None)

        # Check status code
        assert response["statusCode"] == 200

        # Parse JSON body
        body = json.loads(response["body"])
        assert "url" in body

        # Print signed URL for debugging
        print("\nSigned URL:", body["url"])
