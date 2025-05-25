import pytest

from polypy import (
    CHAIN_ID,
    ENDPOINT,
    SIDE,
    TRADE_STATUS,
    MarketIdQuintet,
    MarketIdTriplet,
    PositionManager,
    RPCSettings,
    dec,
)
from polypy.exceptions import (
    PolyPyException,
    PositionTrackingException,
    PositionTransactionException,
)

# noinspection PyProtectedMember
from polypy.manager.rpc_ops import (
    MTX,
    _tx_pre_batch_operate_positions,
    tx_redeem_positions,
)

# noinspection PyProtectedMember
from polypy.manager.rpc_proc import (
    _assert_redeem_sizes_onchain,
    _tx_pre_convert_position,
    _tx_pre_merge_positions,
    _tx_pre_redeem_positions,
    _tx_pre_split_position,
)


# noinspection DuplicatedCode
@pytest.fixture(scope="function")
def position_manager() -> PositionManager:
    return PositionManager("", "", dec(100))


@pytest.fixture(scope="function")
def rpc_settings(private_key) -> RPCSettings:
    return RPCSettings(
        "",
        "",
        CHAIN_ID.AMOY,
        private_key,
        None,
        1,
        None,
        True,
        None,
        None,
        None,
        None,
        None,
    )


@pytest.fixture(scope="function")
def market_triplet() -> MarketIdTriplet:
    return MarketIdTriplet("0x000000", "000001", "000002")


def test_pre_split_position(position_manager, rpc_settings, market_triplet):
    cond_id, amount, neg_risk = _tx_pre_split_position(
        position_manager, market_triplet, dec(11.5), False, ""
    )
    assert cond_id == "0x000000"
    assert amount == dec("11.5")
    assert neg_risk is False
    assert position_manager.balance_available == dec(100)
    assert len(position_manager.asset_ids) == 1

    # overspending
    with pytest.raises(PositionTransactionException):
        _tx_pre_split_position(position_manager, market_triplet, dec(100.01), False, "")


# noinspection DuplicatedCode
def test_pre_merge_positions(position_manager, rpc_settings, market_triplet):
    position_manager.transact(
        market_triplet[1], dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        market_triplet[2], dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    cond_id, size, neg_risk = _tx_pre_merge_positions(
        position_manager, market_triplet, dec(9), False, ""
    )
    assert cond_id == market_triplet[0]
    assert size == dec(9)
    assert neg_risk is False
    assert position_manager.balance_available == dec(100)
    assert position_manager.get_by_id(market_triplet[1]).size == dec(10)
    assert position_manager.get_by_id(market_triplet[2]).size == dec(12)
    assert len(position_manager.asset_ids) == 3

    # size None: auto-infer
    cond_id, size, neg_risk = _tx_pre_merge_positions(
        position_manager, market_triplet, None, False, ""
    )
    assert cond_id == market_triplet[0]
    assert size == dec(10)
    assert neg_risk is False
    assert position_manager.balance_available == dec(100)
    assert position_manager.get_by_id(market_triplet[1]).size == dec(10)
    assert position_manager.get_by_id(market_triplet[2]).size == dec(12)
    assert len(position_manager.asset_ids) == 3

    # overspending
    with pytest.raises(PositionTransactionException):
        _tx_pre_merge_positions(
            position_manager, market_triplet, dec(12.0001), False, ""
        )
    assert position_manager.balance_available == dec(100)
    assert position_manager.get_by_id(market_triplet[1]).size == dec(10)
    assert position_manager.get_by_id(market_triplet[2]).size == dec(12)
    assert len(position_manager.asset_ids) == 3


# noinspection DuplicatedCode
def test_pre_convert_positions(position_manager, rpc_settings):
    position_manager.transact(
        "N1", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(3), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "Y1", dec(1), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]

    nrm_id, size, q_ids, all_mkt_qts = _tx_pre_convert_position(
        position_manager,
        [
            MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
            MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        ],
        all_market_quintets,
        1,
        "",
    )
    assert nrm_id == "0x9"
    assert size == dec(1)
    assert q_ids == ["0x000", "0x001"]
    assert all_mkt_qts == all_market_quintets

    # size None: auto-infer
    nrm_id, size, q_ids, all_mkt_qts = _tx_pre_convert_position(
        position_manager,
        [
            MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        ],
        all_market_quintets,
        None,
        "",
    )
    assert nrm_id == "0x9"
    assert size == dec(12)
    assert q_ids == ["0x000"]
    assert all_mkt_qts == all_market_quintets

    assert position_manager.balance_available == dec(100)
    assert position_manager.get_by_id("N1").size == dec(12)
    assert position_manager.get_by_id("N2").size == dec(3)
    assert position_manager.get_by_id("Y1").size == dec(1)
    assert len(position_manager.asset_ids) == 4


# noinspection DuplicatedCode
def test_pre_convert_positions_raises(position_manager, rpc_settings):
    # failure cases:
    #   overspending size,
    #   all_market_quintets with gaps,
    #   cvt_market_quintet > all_market_quintets,
    #   no position in cvt_market_quintet,
    #   empty all_market_quintets,
    #   non-negative risk markets,
    #   mixed neg risk market id

    position_manager.transact(
        "N1", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(3), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "Y1", dec(1), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]

    # overspending
    with pytest.raises(PositionTransactionException):
        _tx_pre_convert_position(
            position_manager,
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
                MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
            ],
            all_market_quintets,
            dec(12.001),
            "",
        )

    # all_market_quintets with gaps
    with pytest.raises(PolyPyException):
        _tx_pre_convert_position(
            position_manager,
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
                MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
            ],
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
                MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
            ],
            dec(1),
            "",
        )

    # cvt_market_quintet > all_market_quintets
    with pytest.raises(PositionTransactionException) as record:
        _tx_pre_convert_position(
            position_manager,
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
                MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
                MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4"),
            ],
            all_market_quintets,
            dec(1),
            "",
        )
    assert "Missed" in str(record)

    # no position in cvt_market_quintet
    with pytest.raises(PositionTransactionException):
        _tx_pre_convert_position(
            position_manager,
            [MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3")],
            all_market_quintets,
            dec(2),
            "",
        )

    # empty all_market_quintets
    with pytest.raises(PolyPyException):
        _tx_pre_convert_position(
            position_manager,
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
                MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
            ],
            [],
            dec(2),
            "",
        )

    # non-negative risk markets
    with pytest.raises(PositionTransactionException) as record:
        _tx_pre_convert_position(
            position_manager,
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
            ],
            all_market_quintets + [MarketIdQuintet("0x22", "", "0x003", "Y22", "N22")],
            dec(2),
            "",
        )
    assert "Not all 'market_quintets' are NegRiskMarket!" in str(record)

    # mixed neg risk market id
    with pytest.raises(PositionTransactionException) as record:
        _tx_pre_convert_position(
            position_manager,
            [
                MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
            ],
            all_market_quintets
            + [MarketIdQuintet("0x22", "0x5", "0x003", "Y22", "N22")],
            dec(2),
            "",
        )
    assert "Not all 'market_quintets' have the same neg_risk_market_id." in str(record)

    assert position_manager.balance_available == dec(100)
    assert position_manager.get_by_id("N1").size == dec(12)
    assert position_manager.get_by_id("N2").size == dec(3)
    assert position_manager.get_by_id("Y1").size == dec(1)
    assert len(position_manager.asset_ids) == 4


def test_pre_redeem_positions(position_manager, rpc_settings):
    size_1, size_2 = _tx_pre_redeem_positions(
        position_manager, MarketIdTriplet("0x1", "Y1", "N1")
    )
    assert size_1 == 0
    assert size_2 == 0

    position_manager.transact(
        "Y1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size_1, size_2 = _tx_pre_redeem_positions(
        position_manager, MarketIdTriplet("0x1", "Y1", "N1")
    )
    assert size_1 == dec(10)
    assert size_2 == 0

    position_manager.transact(
        "N1", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size_1, size_2 = _tx_pre_redeem_positions(
        position_manager, MarketIdTriplet("0x1", "Y1", "N1")
    )
    assert size_1 == dec(10)
    assert size_2 == dec(12)


def test_raises_triplet_vs_quintet(position_manager, rpc_settings):
    with pytest.raises(PolyPyException) as record:
        # noinspection PyTypeChecker
        _tx_pre_split_position(
            position_manager,
            MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
            dec(1),
            False,
            "",
        )
    assert "triplet" in str(record)

    with pytest.raises(PolyPyException) as record:
        # noinspection PyTypeChecker
        _tx_pre_merge_positions(
            position_manager,
            MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
            dec(1),
            False,
            "",
        )
    assert "triplet" in str(record)

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    with pytest.raises(PolyPyException) as record:
        # noinspection PyTypeChecker
        _tx_pre_convert_position(
            position_manager,
            MarketIdTriplet("0x1", "Y1", "N1"),
            all_market_quintets,
            dec(1),
            "",
        )
    assert "quintet" in str(record)

    all_market_triplets = [
        MarketIdTriplet("0x1", "Y1", "N1"),
        MarketIdTriplet("0x2", "Y2", "N2"),
        MarketIdTriplet("0x3", "Y3", "N3"),
    ]
    with pytest.raises(PolyPyException) as record:
        # noinspection PyTypeChecker
        _tx_pre_convert_position(
            position_manager,
            all_market_quintets[0],
            all_market_triplets,
            dec(1),
            "",
        )
    assert "quintet" in str(record)

    with pytest.raises(PolyPyException) as record:
        # noinspection PyTypeChecker
        tx_redeem_positions(
            rpc_settings, [position_manager], all_market_quintets[0], "YES", False, ""
        )
    assert "triplet" in str(record)


def test_assert_redeem_sizes_onchain(private_key):
    rpc = RPCSettings(
        ENDPOINT.RPC_POLYGON,
        None,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        1,
        None,
        False,
        None,
        None,
        None,
        None,
        None,
    )
    triplet = MarketIdTriplet(
        "0xd35868afa9257d08411f4c7d61601213fc3d9fa1ebbc1d963572263904be2616",
        "20043121189468219599437255198027106640631597812114545269112896531495574826378",
        "75308226540519873239233623140240170988653500419723135176760582150191967925886",
    )
    _assert_redeem_sizes_onchain(rpc, 0, 0, triplet)

    with pytest.raises(PositionTrackingException) as record:
        _assert_redeem_sizes_onchain(rpc, 1, 0, triplet)
    assert "on-chain" in str(record)

    with pytest.raises(PositionTrackingException) as record:
        _assert_redeem_sizes_onchain(rpc, 1, 1, triplet)
    assert "on-chain" in str(record)


# noinspection DuplicatedCode
def test_pre_batch_operate_positions(position_manager):
    txn = _tx_pre_batch_operate_positions(position_manager, [], "", "")
    assert txn == []

    triplet1 = MarketIdTriplet("0x01", "Y1", "N1")
    triplet2 = MarketIdTriplet("0x02", "Y2", "N2")
    all_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y11", "N11"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y12", "N12"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y13", "N13"),
    ]
    cvt_quintets = all_quintets[:2]
    position_manager.transact(
        "Y2", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N11", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N12", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    mtxs = [
        MTX.split(triplet1, dec(2), False),
        MTX.merge(triplet2, dec(4), False),
        MTX.convert(cvt_quintets, all_quintets, dec(2), False),
    ]

    txn = _tx_pre_batch_operate_positions(position_manager, mtxs, "", "")

    assert txn[0].condition_id == triplet1[0]
    assert txn[0].amount_usdc == dec(2)
    assert txn[1].condition_id == triplet2[0]
    assert txn[1].size == dec(4)
    assert txn[2].neg_risk_market_id == "0x9"
    assert txn[2].size == dec(2)
    assert txn[2].question_ids == ["0x000", "0x001"]
    assert mtxs[2].all_market_quintets == all_quintets
    assert position_manager.balance == dec(100)


# noinspection DuplicatedCode,SpellCheckingInspection
def test_pre_batch_operate_positions_raises(position_manager):
    # failure cases (just test a few to see if _tx_pre ops trigger):
    #   triplet vs quintet, position not in position manager for merge, overspending for merge
    triplet1 = MarketIdTriplet("0x01", "Y1", "N1")
    triplet2 = MarketIdTriplet("0x02", "Y2", "N2")
    all_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y11", "N11"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y12", "N12"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y13", "N13"),
    ]
    cvt_quintets = all_quintets[:2]
    position_manager.transact(
        "N11", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N12", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    # position not in manager for merge
    mtxs = [
        MTX.split(triplet1, dec(2), False),
        MTX.merge(triplet2, dec(4), False),
        MTX.convert(cvt_quintets, all_quintets, dec(2), False),
    ]
    with pytest.raises(PositionTrackingException):
        _tx_pre_batch_operate_positions(position_manager, mtxs, "", "")

    # overspending in merge
    position_manager.transact(
        "Y2", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    mtxs = [
        MTX.split(triplet1, dec(2), False),
        MTX.merge(triplet2, dec(8), False),
        MTX.convert(cvt_quintets, all_quintets, dec(2), False),
    ]
    with pytest.raises(PositionTransactionException):
        _tx_pre_batch_operate_positions(position_manager, mtxs, "", "")

    # triplets in convert
    # noinspection PyTypeChecker
    mtxs = [
        MTX.split(triplet1, dec(2), False),
        MTX.merge(triplet2, dec(8), False),
        MTX.convert(
            MarketIdTriplet.from_market_quintet(all_quintets[0]),
            all_quintets,
            dec(2),
            False,
        ),
    ]
    with pytest.raises(PolyPyException):
        _tx_pre_batch_operate_positions(position_manager, mtxs, "", "")

    assert position_manager.balance == dec(100)


# noinspection DuplicatedCode
def test_pre_batch_operate_positions_write_mtx(position_manager, mocker):
    all_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y11", "N11"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y12", "N12"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y13", "N13"),
    ]
    cvt_quintets = all_quintets[:2]
    position_manager.transact(
        "N11", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N12", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    # mock
    mocker.patch(
        "polypy.manager.rpc_proc.get_neg_risk_markets",
        return_value=(False, all_quintets),
    )

    mtxs = [
        MTX.convert(cvt_quintets, None, dec(2), False),
    ]
    assert mtxs[0].all_market_quintets is None

    _tx_pre_batch_operate_positions(position_manager, mtxs, "", "")

    assert mtxs[0].all_market_quintets == all_quintets
