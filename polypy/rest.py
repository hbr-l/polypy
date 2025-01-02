from functools import lru_cache
from typing import Any

import msgspec
import requests

# todo integrate rate_limiter

END_CURSOR = "LTE="


def _get(url: str) -> requests.Response:
    resp = requests.request("GET", url)
    resp.raise_for_status()
    return resp


def get_orderbook(endpoint: str, asset_id: str) -> dict[str, Any]:
    resp = _get(f"{endpoint}/book?token_id={asset_id}")
    return msgspec.json.decode(resp.text)


def get_tick_size(endpoint: str, asset_id: str) -> float:
    resp = _get(f"{endpoint}/tick-size?token_id={asset_id}")
    return msgspec.json.decode(resp.text)["minimum_tick_size"]


@lru_cache(maxsize=16)
def get_neg_risk(endpoint: str, asset_id: str) -> bool:
    resp = _get(f"{endpoint}/neg-risk?token_id={asset_id}")
    return msgspec.json.decode(resp.text)["neg_risk"]
