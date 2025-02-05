import time

import pytest
from freezegun import freeze_time
from py_clob_client.client import (
    ApiCreds,
    ClobClient,
    OrderArgs,
    RequestArgs,
    Signer,
    create_level_2_headers,
    order_to_json,
)
from py_clob_client.clob_types import OrderType
from py_clob_client.constants import POLYGON
from py_clob_client.endpoints import POST_ORDER
from py_clob_client.order_builder.constants import BUY
from py_clob_client.signing.hmac import build_hmac_signature

from polypy.constants import CHAIN_ID
from polypy.exceptions import PolyPyException
from polypy.order.limit import SIDE, create_limit_order
from polypy.rest.api import build_auth_header
from polypy.signing import SIGNATURE_TYPE
from polypy.signing import build_hmac_signature as poly_build_hmac_signature
from polypy.signing import polymarket_domain


def test_polymarket_domain():
    t_dom = polymarket_domain(CHAIN_ID.POLYGON, True)
    f_dom = polymarket_domain(CHAIN_ID.POLYGON, False)
    assert t_dom != f_dom

    with pytest.raises(PolyPyException):
        # noinspection PyTypeChecker
        polymarket_domain(CHAIN_ID.POLYGON, None)


def test_build_hmac_signature(api_key, secret):
    timestamp = int(time.time())

    clob_hmac = build_hmac_signature(secret, str(timestamp), "POST", "/order")
    poly_hmac = poly_build_hmac_signature(secret, timestamp, "POST", "/order", None)

    assert clob_hmac == poly_hmac


@freeze_time("2024-12-19 17:05:55")
def test_build_auth_header(
    private_key, local_host_addr, fix_seed, api_key, passphrase, secret
):
    token_id = "123"
    price = 0.1
    size = 20

    signer = Signer(private_key, POLYGON)
    creds = ApiCreds(api_key, secret, passphrase)
    client = ClobClient(local_host_addr, POLYGON, private_key, creds)
    client.get_tick_size = lambda x: "0.01"
    client.get_neg_risk = lambda x: False

    # noinspection PyTypeChecker
    clob_order = fix_seed(client.create_order)(OrderArgs(token_id, price, size, BUY))

    body = order_to_json(clob_order, creds.api_key, OrderType.GTC)
    req_args = RequestArgs(method="POST", request_path=POST_ORDER, body=body)
    clob_header = create_level_2_headers(signer, creds, req_args)

    poly_order = fix_seed(create_limit_order)(
        price,
        size,
        token_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    poly_header = build_auth_header(
        private_key,
        api_key,
        secret,
        passphrase,
        "POST",
        "/order",
        poly_order.to_payload(api_key),
    )

    assert clob_order.dict() == poly_order.to_dict()
    assert POST_ORDER == "/order"

    assert body == poly_order.to_payload(api_key)
    assert clob_header == poly_header
