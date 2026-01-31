import os
import requests

DATASET_ID = "d6yy-54nr"
ENDPOINT = f"https://data.ny.gov/resource/{DATASET_ID}.json"


def test_socrata_api_up():
    """
    Basic health check:
    - endpoint responds 200
    - returns a JSON list with at least 1 row
    """
    headers = {}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token

    r = requests.get(ENDPOINT, params={"$limit": 5}, headers=headers, timeout=20)
    assert r.status_code == 200, f"Unexpected status: {r.status_code} body={r.text[:200]}"
    data = r.json()

    assert isinstance(data, list)
    assert len(data) > 0


def test_expected_fields_present():
    """
    The dataset should contain these fields for analysis.
    We only check that at least one row includes the keys.
    """
    r = requests.get(ENDPOINT, params={"$limit": 50}, timeout=20)
    assert r.status_code == 200
    data = r.json()

    assert isinstance(data, list)
    assert len(data) > 0

    # Not all rows must contain all fields, but at least one should.
    required = {"draw_date", "winning_numbers"}
    found = any(required.issubset(set(row.keys())) for row in data)
    assert found, f"Required keys {required} not found in sample rows"
