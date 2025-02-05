import math

import pytest

from polypy.constants import CHAIN_ID
from polypy.order import SIDE, create_limit_order, create_market_order
from polypy.order.limit import limit_order_taker_maker_amount
from polypy.order.market import market_order_taker_maker_amount
from polypy.rounding import max_allowed_decimals, round_half_even
from polypy.signing import SIGNATURE_TYPE

#######################################
# Port tests from py_clob_client
# (tests for polypy package see test_order.py)
#######################################


@pytest.mark.skip(
    reason="py-clob-client computes marketable buy price with descending order of open asks, "
    "but ascending order like in polypy would make more sense."
)
def test_calculate_market_price_buy():
    # todo check py-clob-client for updates:
    #   https://github.com/Polymarket/py-clob-client/issues/106
    pass


@pytest.mark.skip(
    reason="py-clob-client computes marketable sell price with ascending order of open bids, "
    "but descending order like in polypy would make more sense."
)
def test_calculate_market_price_sell():
    # todo check py-clob-client for updates:
    #   https://github.com/Polymarket/py-clob-client/issues/106
    pass


@pytest.mark.parametrize(
    "delta_price, delta_size, size_init, max_size, price_init, tick_size_digits, round_digits",
    [
        # test_get_order_amounts_buy_0_1, test_get_order_amounts_sell_0_1
        (0.1, 0.01, 0.01, 1000, 0.1, 1, 2),
        # test_get_order_amounts_buy_0_01, test_get_order_amounts_sell_0_01
        (0.01, 0.01, 0.01, 100, 0.01, 2, 4),
        # test_get_order_amounts_buy_0_001, test_get_order_amounts_sell_0_001
        (0.001, 0.01, 0.01, 10, 0.001, 3, 6),
        # test_get_order_amounts_buy_0_0001, test_get_order_amounts_sell_0_0001
        (0.0001, 0.01, 0.01, 1, 0.0001, 4, 8),
    ],
)
def test_limit_order_amount(
    delta_price,
    delta_size,
    size_init,
    max_size,
    price_init,
    tick_size_digits,
    round_digits,
    patch_py_clob_client_rounding,
):
    # port: py-clob-client test_builder.test_get_order_amounts_buy_0_1
    # port: py-clob-client test_builder.test_get_order_amounts_sell_0_1

    # port: py-clob-client test_builder.test_get_order_amounts_buy_0_01
    # port: py-clob-client test_builder.test_get_order_amounts_sell_0_01

    # port: py-clob-client test_builder.test_get_order_amounts_buy_0_001
    # port: py-clob-client test_builder.test_get_order_amounts_sell_0_001

    # port: py-clob-client test_builder.test_get_order_amounts_buy_0_0001
    # port: py-clob-client test_builder.test_get_order_amounts_sell_0_0001

    size = size_init
    while size <= max_size:
        price = price_init
        while price <= 1:
            maker, taker = limit_order_taker_maker_amount(
                SIDE.BUY, price, size, tick_size_digits, 2
            )
            assert max_allowed_decimals(maker, 0)
            assert max_allowed_decimals(taker, 0)
            assert round_half_even(maker / taker, round_digits) >= round_half_even(
                price, round_digits
            )

            maker, taker = limit_order_taker_maker_amount(
                SIDE.SELL, price, size, tick_size_digits, 2
            )
            assert max_allowed_decimals(maker, 0)
            assert max_allowed_decimals(taker, 0)
            assert round_half_even(taker / maker, round_digits) >= round_half_even(
                price, round_digits
            )

            price += delta_price

        size += delta_size


@pytest.mark.parametrize(
    "delta_price, delta_size, amount_init, price_init, max_amount, tick_size_digits, round_digits",
    [
        (0.1, 0.01, 0.01, 0.1, 1000, 1, 2),  # test_get_market_order_amounts_0_1
        (0.01, 0.01, 0.01, 0.01, 100, 2, 4),  # test_get_market_order_amounts_0_01
        (0.001, 0.01, 0.01, 0.001, 10, 3, 6),  # test_get_market_order_amounts_0_001
        (0.0001, 0.01, 0.01, 0.0001, 1, 4, 8),  # test_get_market_order_amounts_0_0001
    ],
)
def test_market_order_amount(
    delta_price,
    delta_size,
    amount_init,
    price_init,
    max_amount,
    tick_size_digits,
    round_digits,
    patch_py_clob_client_rounding,
):
    # port: py-clob-client test_builder.test_get_market_order_amounts_0_1
    # port: py-clob-client test_builder.test_get_market_order_amounts_0_01
    # port: py-clob-client test_builder.test_get_market_order_amounts_0_001
    # port: py-clob-client test_builder.test_get_market_order_amounts_0_0001

    amount = amount_init
    while amount <= max_amount:
        price = price_init
        while price <= 1:
            maker, taker = market_order_taker_maker_amount(
                SIDE.BUY, amount, price, tick_size_digits, 2
            )
            assert max_allowed_decimals(maker, 0)
            assert max_allowed_decimals(taker, 0)
            assert round_half_even(maker / taker, round_digits) >= round_half_even(
                price, round_digits
            )

            # not in py-clob-client
            maker, taker = market_order_taker_maker_amount(
                SIDE.SELL, amount, price, tick_size_digits, 2, max_size=math.inf
            )
            assert max_allowed_decimals(maker, 0)
            assert max_allowed_decimals(taker, 0)
            assert round_half_even(taker / maker, round_digits) >= round_half_even(
                price, round_digits
            )

            price += delta_price

        amount += delta_size


# todo parametrize
@pytest.mark.parametrize("neg_risk", [False] * 9 + [True] * 9)
def test_create_limit_order_decimal_accuracy(
    neg_risk, patch_py_clob_client_rounding, private_key
):
    # port: py-clob-client test_builder.test_create_order_decimal_accuracy
    # port: py-clob-client test_builder.test_create_order_decimal_accuracy_neg_risk

    buy_order = create_limit_order(
        0.24,
        15,
        "123",
        SIDE.BUY,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    buy_order_dict = buy_order.to_dict()
    assert buy_order_dict["makerAmount"] == "3600000"
    assert buy_order_dict["takerAmount"] == "15000000"

    sell_order = create_limit_order(
        0.24,
        15,
        "123",
        SIDE.SELL,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    sell_order_dict = sell_order.to_dict()
    assert sell_order_dict["makerAmount"] == "15000000"
    assert sell_order_dict["takerAmount"] == "3600000"

    buy_order = create_limit_order(
        0.82,
        101,
        "123",
        SIDE.BUY,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    buy_order_dict = buy_order.to_dict()
    assert buy_order_dict["makerAmount"] == "82820000"
    assert buy_order_dict["takerAmount"] == "101000000"

    sell_order = create_limit_order(
        0.82,
        101,
        "123",
        SIDE.SELL,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    sell_order_dict = sell_order.to_dict()
    assert sell_order_dict["makerAmount"] == "101000000"
    assert sell_order_dict["takerAmount"] == "82820000"

    buy_order = create_limit_order(
        0.78,
        12.8205,
        "123",
        SIDE.BUY,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    buy_order_dict = buy_order.to_dict()
    assert buy_order_dict["makerAmount"] == "9999600"
    assert buy_order_dict["takerAmount"] == "12820000"

    sell_order = create_limit_order(
        0.78,
        12.8205,
        "123",
        SIDE.SELL,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    sell_order_dict = sell_order.to_dict()
    assert sell_order_dict["makerAmount"] == "12820000"
    assert sell_order_dict["takerAmount"] == "9999600"

    sell_order = create_limit_order(
        0.39,
        2435.89,
        "123",
        SIDE.SELL,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    sell_order_dict = sell_order.to_dict()
    assert sell_order_dict["makerAmount"] == "2435890000"
    assert sell_order_dict["takerAmount"] == "949997100"

    sell_order = create_limit_order(
        0.43,
        19.1,
        "123",
        SIDE.SELL,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    sell_order_dict = sell_order.to_dict()
    assert sell_order_dict["makerAmount"] == "19100000"
    assert sell_order_dict["takerAmount"] == "8213000"

    buy_order = create_limit_order(
        0.58,
        18233.33,
        "123",
        SIDE.BUY,
        0.01,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    buy_order_dict = buy_order.to_dict()
    assert buy_order_dict["makerAmount"] == "10575331400"
    assert buy_order_dict["takerAmount"] == "18233330000"
    assert (
        int(buy_order_dict["makerAmount"]) / int(buy_order_dict["takerAmount"]) == 0.58
    )


# noinspection DuplicatedCode
@pytest.mark.parametrize(
    "price, size, token_id, side, tick_size, neg_risk, fee_rate_bps, nonce, expiration, "
    "maker, signer, taker, maker_amount, taker_amount, signature_type",
    [
        (  # test_create_order_buy_0_1
            0.5,
            21.04,
            "123",
            SIDE.BUY,
            0.1,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            10520000,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_buy_0_01, test_dict_order_buy
            0.56,
            21.04,
            "123",
            SIDE.BUY,
            0.01,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            11782400,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_buy_0_001
            0.056,
            21.04,
            "123",
            SIDE.BUY,
            0.001,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            1178240,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_buy_0_0001
            0.0056,
            21.04,
            "123",
            SIDE.BUY,
            0.0001,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            117824,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_sell_0_1
            0.5,
            21.04,
            "123",
            SIDE.SELL,
            0.1,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            10520000,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_sell_0_01, test_dict_order_sell
            0.56,
            21.04,
            "123",
            SIDE.SELL,
            0.01,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            11782400,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_sell_0_001
            0.056,
            21.04,
            "123",
            SIDE.SELL,
            0.001,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            1178240,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_sell_0_0001
            0.0056,
            21.04,
            "123",
            SIDE.SELL,
            0.0001,
            False,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            117824,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_buy_0_1_neg_risk
            0.5,
            21.04,
            "123",
            SIDE.BUY,
            0.1,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            10520000,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_buy_0_01_neg_risk, test_dict_order_buy_neg_risk
            0.56,
            21.04,
            "123",
            SIDE.BUY,
            0.01,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            11782400,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_buy_0_001_neg_risk
            0.056,
            21.04,
            "123",
            SIDE.BUY,
            0.001,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            1178240,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_buy_0_0001_neg_risk
            0.0056,
            21.04,
            "123",
            SIDE.BUY,
            0.0001,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            117824,
            21040000,
            SIGNATURE_TYPE.EOA,
        ),
        (  # test_create_order_sell_0_1_neg_risk
            0.5,
            21.04,
            "123",
            SIDE.SELL,
            0.1,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            10520000,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_sell_0_01_neg_risk, test_dict_order_sell_neg_risk
            0.56,
            21.04,
            "123",
            SIDE.SELL,
            0.01,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            11782400,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_sell_0_001_neg_risk
            0.056,
            21.04,
            "123",
            SIDE.SELL,
            0.001,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            1178240,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
        (  # test_create_order_sell_0_0001_neg_risk
            0.0056,
            21.04,
            "123",
            SIDE.SELL,
            0.0001,
            True,
            111,
            123,
            50000,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            21040000,
            117824,
            SIGNATURE_TYPE.POLY_GNOSIS_SAFE,
        ),
    ],
)
def test_create_limit_order_buy_sell_0_x(
    price,
    size,
    token_id,
    side,
    tick_size,
    neg_risk,
    fee_rate_bps,
    nonce,
    expiration,
    maker,
    signer,
    taker,
    maker_amount,
    taker_amount,
    signature_type,
    patch_py_clob_client_rounding,
    private_key,
):
    # port: py-clob-client test_builder.test_create_order_buy_0_1
    # port: py-clob-client test_builder.test_create_order_buy_0_01
    # port: py-clob-client test_builder.test_create_order_buy_0_001
    # port: py-clob-client test_builder.test_create_order_buy_0_0001

    # port: py-clob-client test_builder.test_create_order_sell_0_1
    # port: py-clob-client test_builder.test_create_order_sell_0_01
    # port: py-clob-client test_builder.test_create_order_sell_0_001
    # port: py-clob-client test_builder.test_create_order_sell_0_0001

    # port: py-clob-client test_builder.test_create_order_buy_0_1_neg_risk
    # port: py-clob-client test_builder.test_create_order_buy_0_01_neg_risk
    # port: py-clob-client test_builder.test_create_order_buy_0_001_neg_risk
    # port: py-clob-client test_builder.test_create_order_buy_0_0001_neg_risk

    # port: py-clob-client test_builder.test_create_order_sell_0_1_neg_risk
    # port: py-clob-client test_builder.test_create_order_sell_0_01_neg_risk
    # port: py-clob-client test_builder.test_create_order_sell_0_001_neg_risk
    # port: py-clob-client test_builder.test_create_order_sell_0_0001_neg_risk

    # port: py-clob-client test_builder.test_dict_order_buy_neg_risk
    # port: py-clob-client test_builder.test_dict_order_sell_neg_risk

    # port: py-clob-client test_builder.test_dict_order_buy
    # port: py-clob-client test_builder.test_dict_order_sell

    order = create_limit_order(
        price,
        size,
        token_id,
        side,
        tick_size,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        maker=None,
        signature_type=signature_type,
        expiration=expiration,
        nonce=nonce,
        fee_rate_bps=fee_rate_bps,
    )
    order_dict = order.to_dict()

    # -----------------
    # test against dict
    # -----------------
    assert isinstance(order_dict["salt"], int)
    assert order_dict["maker"] == maker
    assert order_dict["signer"] == signer
    assert order_dict["taker"] == taker
    assert order_dict["tokenId"] == token_id
    assert order_dict["makerAmount"] == str(maker_amount)
    assert order_dict["takerAmount"] == str(taker_amount)
    assert order_dict["side"] == str(side)
    assert order_dict["expiration"] == str(expiration)
    assert order_dict["nonce"] == str(nonce)
    assert order_dict["feeRateBps"] == str(fee_rate_bps)
    assert order_dict["signatureType"] == int(signature_type)
    assert order_dict["signature"] is not None

    # select maker or taker amount depending on side
    numerator = int(order_dict["makerAmount"]) * (side == SIDE.BUY) + int(
        order_dict["takerAmount"]
    ) * (side == SIDE.SELL)
    denominator = int(order_dict["takerAmount"]) * (side == SIDE.BUY) + int(
        order_dict["makerAmount"]
    ) * (side == SIDE.SELL)
    round_digits = -int(math.log10(tick_size)) * 2
    assert round_half_even(numerator / denominator, round_digits) == price

    # ------------------------------
    # test against object attributes
    # ------------------------------
    assert isinstance(order.salt, int)
    assert order.maker == maker
    assert order.signer == signer
    assert order.taker == taker
    assert order.token_id == token_id
    assert order.maker_amount == maker_amount
    assert order.taker_amount == taker_amount
    assert order.side == side
    assert order.expiration == expiration
    assert order.nonce == nonce
    assert order.fee_rate_bps == fee_rate_bps
    assert order.signature_type == signature_type
    assert order.signature is not None

    # select maker or taker amount depending on side
    numerator = order.maker_amount * (side == SIDE.BUY) + order.taker_amount * (
        side == SIDE.SELL
    )
    denominator = order.taker_amount * (side == SIDE.BUY) + order.maker_amount * (
        side == SIDE.SELL
    )
    assert round_half_even(numerator / denominator, round_digits) == price


# noinspection DuplicatedCode
@pytest.mark.parametrize(
    "token_id, price, amount, fee_rate_bps, nonce, tick_size, neg_risk, maker, "
    "signer, taker, maker_amount, taker_amount, signature_type, side",
    [
        (  # test_create_market_order_buy_0_1
            "123",
            0.5,
            100,
            111,
            123,
            0.1,
            False,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            200000000,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_01
            "123",
            0.56,
            100,
            111,
            123,
            0.01,
            False,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            178571400,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_001
            "123",
            0.056,
            100,
            111,
            123,
            0.001,
            False,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            1785714280,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_0001
            "123",
            0.0056,
            100,
            111,
            123,
            0.0001,
            False,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            17857142857,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_1_neg_risk
            "123",
            0.5,
            100,
            111,
            123,
            0.1,
            True,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            200000000,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_01_neg_risk
            "123",
            0.56,
            100,
            111,
            123,
            0.01,
            True,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            178571400,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_001_neg_risk
            "123",
            0.056,
            100,
            111,
            123,
            0.001,
            True,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            1785714280,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
        (  # test_create_market_order_buy_0_0001_neg_risk
            "123",
            0.0056,
            100,
            111,
            123,
            0.0001,
            True,
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "0x0000000000000000000000000000000000000000",
            100000000,
            17857142857,
            SIGNATURE_TYPE.EOA,
            SIDE.BUY,
        ),
    ],
)
def test_create_market_order_buy_0_x(
    token_id,
    price,
    amount,
    fee_rate_bps,
    nonce,
    tick_size,
    neg_risk,
    maker,
    signer,
    taker,
    maker_amount,
    taker_amount,
    signature_type,
    side,
    patch_py_clob_client_rounding,
    private_key,
):
    # py-clob-client does not define a market sell order (+ has bugs in its rounding approach),
    #   but we will later on test market sell order separately

    # port: py-clob-client test_builder.test_create_market_order_buy_0_1
    # port: py-clob-client test_builder.test_create_market_order_buy_0_01
    # port: py-clob-client test_builder.test_create_market_order_buy_0_001
    # port: py-clob-client test_builder.test_create_market_order_buy_0_0001

    # port: py-clob-client test_builder.test_create_market_order_buy_0_1_neg_risk
    # port: py-clob-client test_builder.test_create_market_order_buy_0_01_neg_risk
    # port: py-clob-client test_builder.test_create_market_order_buy_0_001_neg_risk
    # port: py-clob-client test_builder.test_create_market_order_buy_0_0001_neg_risk

    order = create_market_order(
        amount,
        token_id,
        SIDE.BUY,
        tick_size,
        neg_risk,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        nonce=nonce,
        fee_rate_bps=fee_rate_bps,
        price=price,
    )
    order_dict = order.to_dict()

    # -----------------
    # test against dict
    # -----------------
    assert isinstance(order_dict["salt"], int)
    assert order_dict["maker"] == maker
    assert order_dict["signer"] == signer
    assert order_dict["taker"] == taker
    assert order_dict["tokenId"] == token_id
    assert order_dict["makerAmount"] == str(maker_amount)
    assert order_dict["takerAmount"] == str(taker_amount)
    assert order_dict["side"] == str(side)
    assert order_dict["expiration"] == str(0)
    assert order_dict["nonce"] == str(nonce)
    assert order_dict["feeRateBps"] == str(fee_rate_bps)
    assert order_dict["signatureType"] == int(signature_type)
    assert order_dict["signature"] is not None

    # select maker or taker amount depending on side
    numerator = int(order_dict["makerAmount"]) * (side == SIDE.BUY) + int(
        order_dict["takerAmount"]
    ) * (side == SIDE.SELL)
    denominator = int(order_dict["takerAmount"]) * (side == SIDE.BUY) + int(
        order_dict["makerAmount"]
    ) * (side == SIDE.SELL)
    round_digits = -int(math.log10(tick_size)) * 2
    assert round_half_even(numerator / denominator, round_digits) == price

    # round_half_even(maker / taker, round_digits) >= round_half_even(
    #                 price, round_digits
    #             )

    # ------------------------------
    # test against object attributes
    # ------------------------------
    assert isinstance(order.salt, int)
    assert order.maker == maker
    assert order.signer == signer
    assert order.taker == taker
    assert order.token_id == token_id
    assert order.maker_amount == maker_amount
    assert order.taker_amount == taker_amount
    assert order.side == side
    assert order.expiration == 0
    assert order.nonce == nonce
    assert order.fee_rate_bps == fee_rate_bps
    assert order.signature_type == signature_type
    assert order.signature is not None

    # select maker or taker amount depending on side
    numerator = order.maker_amount * (side == SIDE.BUY) + order.taker_amount * (
        side == SIDE.SELL
    )
    denominator = order.taker_amount * (side == SIDE.BUY) + order.maker_amount * (
        side == SIDE.SELL
    )
    assert round_half_even(numerator / denominator, round_digits) == price
