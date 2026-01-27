import sys
import pytest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "lambda"))

from proxy.handler import sign_url, validate_user, s3_parse_s3_url

# ---------------------------------
# validate_user
# ---------------------------------
def test_validate_user():
    assert validate_user("Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem") is True
    assert validate_user("wrong") is False

# ---------------------------------
# s3_parse_s3_url
# ---------------------------------
def test_s3_parse_s3_url():
    bucket, key = s3_parse_s3_url("s3://mybucket/path/to/file.txt")
    assert bucket == "mybucket"
    assert key == "path/to/file.txt"

    bucket, key = s3_parse_s3_url("s3://mybucket")
    assert bucket == "mybucket"
    assert key == ""

    with pytest.raises(ValueError):
        s3_parse_s3_url("http://mybucket/file.txt")

# ---------------------------------
# sign_url
# ---------------------------------
def make_event(auth="Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem", s3url="s3://gaia-tsf-private/file.txt"):
    return {
        "headers": {"Authorization": auth},
        "queryStringParameters": {"s3url": s3url}
    }

@patch("proxy.handler.s3.generate_presigned_url")
def test_sign_url_success(mock_presign):
    mock_presign.return_value = "https://signed-url"
    event = make_event()
    response = sign_url(event, None)
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["url"] == "https://signed-url"

def test_sign_url_unauthorized():
    event = make_event(auth="wrong")
    response = sign_url(event, None)
    assert response["statusCode"] == 403
    body = json.loads(response["body"])
    assert "error" in body

def test_sign_url_wrong_bucket():
    event = make_event(s3url="s3://other-bucket/file.txt")
    response = sign_url(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "Requested bucket" in body["error"]

def test_sign_url_missing_key():
    event = make_event(s3url="s3://gaia-tsf-private/")
    response = sign_url(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "Missing 'file'" in body["error"]
