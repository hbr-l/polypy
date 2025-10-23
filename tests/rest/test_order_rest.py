"""Order related REST actions"""
import pytest
from freezegun import freeze_time
from py_clob_client.client import ApiCreds, ClobClient, OrderArgs
from py_clob_client.constants import POLYGON
from py_clob_client.order_builder.constants import BUY

from polypy.constants import CHAIN_ID
from polypy.order.limit import SIDE, create_limit_order
from polypy.rest.api import _overload_headers, build_auth_header
from polypy.signing import SIGNATURE_TYPE


@pytest.fixture
def return_clob_request_params(monkeypatch):
    from py_clob_client.http_helpers.helpers import overloadHeaders

    def _intercept(endpoint: str, method: str, headers=None, data=None):
        headers = overloadHeaders(method, headers)

        if "fee-rate" in endpoint:
            return {"base_fee": 0}
        return endpoint, method, headers, data

    monkeypatch.setattr("py_clob_client.http_helpers.helpers.request", _intercept)


@freeze_time("2024-12-19 17:05:55")
def test_equiv_clob_poly_post_order(
    private_key,
    local_host_addr,
    fix_seed,
    return_clob_request_params,
    api_key,
    passphrase,
    secret,
):
    token_id = "123"
    price = 0.1
    size = 20

    creds = ApiCreds(api_key, secret, passphrase)
    client = ClobClient(local_host_addr, POLYGON, private_key, creds)
    client.get_tick_size = lambda x: "0.01"
    client.get_neg_risk = lambda x: False

    # noinspection PyTypeChecker
    clob_order = fix_seed(client.create_order)(OrderArgs(token_id, price, size, BUY))
    clob_url, clob_method, clob_headers, clob_data = client.post_order(clob_order)

    # PolyPy
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

    def poly_post_args(
        endpoint: str,
        order,
        private_key_: str,
        api_key_: str,
        secret_: str,
        passphrase_: str,
    ):
        body = order.to_payload(api_key_)
        url = f"{endpoint}/order"
        method = "POST"
        headers = build_auth_header(
            private_key_, api_key_, secret_, passphrase_, "POST", "/order", body
        )

        headers = _overload_headers(headers, method)
        return url, method, headers, body

    poly_url, poly_method, poly_headers, poly_data = poly_post_args(
        local_host_addr, poly_order, private_key, api_key, secret, passphrase
    )

    del clob_headers["User-Agent"]

    assert clob_url == poly_url
    assert clob_method == poly_method
    assert clob_headers == poly_headers
    assert clob_data == poly_data
