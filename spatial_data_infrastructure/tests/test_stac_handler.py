import sys
import pytest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "lambda"))

from stac.handler import stac

# Helper for event
def make_event(method="GET", path="/items", query=None, headers=None):
    return {
        "requestContext": {"http": {"method": method}},
        "rawPath": path,
        "queryStringParameters": query or {},
        "headers": headers or {},
    }

# ---------------------------------
# Test forbidden methods
# ---------------------------------
def test_stac_forbidden_method():
    event = make_event(method="POST")
    response = stac(event, None)
    assert response["statusCode"] == 403
    assert response["body"] == "Forbidden"

# ---------------------------------
# Test proxy GET request (mocked)
# ---------------------------------
@patch("stac.handler.requests.request")
def test_stac_get_request(mock_request):
    # Simulating STAC response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.text = '{"features": []}'
    mock_request.return_value = mock_resp

    event = make_event(method="GET", path="/collections")
    response = stac(event, None)

    # Check response
    assert response["statusCode"] == 200
    assert response["headers"]["Content-Type"] == "application/json"
    assert response["body"] == '{"features": []}'

    # Check the call
    mock_request.assert_called_once_with(
        "GET",
        "http://ip-172-31-19-31.eu-central-1.compute.internal:8080/collections",
        headers={},
        params={}
    )

# ---------------------------------
# Test HEAD request
# ---------------------------------
@patch("stac.handler.requests.request")
def test_stac_head_request(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.text = ""
    mock_request.return_value = mock_resp

    event = make_event(method="HEAD", path="/status")
    response = stac(event, None)
    assert response["statusCode"] == 200
    mock_request.assert_called_once_with(
        "HEAD",
        "http://ip-172-31-19-31.eu-central-1.compute.internal:8080/status",
        headers={},
        params={}
    )
