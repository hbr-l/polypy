import threading
import time

import pytest

from polypy import AugmentedConversionCache, MarketIdQuintet, PolyPyException, dec


@pytest.fixture(scope="function")
def conversion_cache() -> AugmentedConversionCache:
    return AugmentedConversionCache("", 10)


def test_update_new(conversion_cache):
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    resize, new_quintets = conversion_cache.update(10, all_market_quintets)

    # if new: we do not need to resize any positions
    assert resize == dec(0)
    assert new_quintets == []
    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        m[0] for m in all_market_quintets
    }
    assert conversion_cache.caches["0x9"].cumulative_size == dec(10)


def test_update_existing(conversion_cache):
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    add_quintet = MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4")
    add_quintet2 = MarketIdQuintet("0x5", "0x9", "0x004", "Y5", "N5")

    resize, new_quintets = conversion_cache.update(10, all_market_quintets)
    resize2, new_quintets2 = conversion_cache.update(
        12,
        all_market_quintets + [add_quintet],
    )
    resize3, new_quintets3 = conversion_cache.update(
        1,
        all_market_quintets + [add_quintet],
    )
    resize4, new_quintets4 = conversion_cache.update(
        1, all_market_quintets + [add_quintet]
    )
    resize5, new_quintets5 = conversion_cache.update(
        None, all_market_quintets + [add_quintet]
    )
    resize6, new_quintets6 = conversion_cache.update(
        None, all_market_quintets + [add_quintet, add_quintet2]
    )

    assert resize == dec(0)
    assert resize2 == dec(10)
    assert resize3 == dec(0)
    assert resize4 == dec(0)
    assert resize5 == dec(0)
    assert resize6 == dec(24)
    assert new_quintets == []
    assert new_quintets2 == [add_quintet[3]]
    assert new_quintets3 == []
    assert new_quintets4 == []
    assert new_quintets5 == []
    assert new_quintets6 == [add_quintet2[3]]
    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        m[0] for m in all_market_quintets + [add_quintet, add_quintet2]
    }
    assert conversion_cache.caches["0x9"].cumulative_size == dec(24)


def test_update_new_raises_max_size(conversion_cache):
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    resize, new_quintets = conversion_cache.update(10, all_market_quintets)

    with pytest.raises(PolyPyException):
        for i in range(20):
            market_quintets = [
                MarketIdQuintet(
                    f"0x{4+i}", f"0x{10+i}", f"0x00{3+i}", f"Y{4+i}", f"N{4+i}"
                )
            ]
            conversion_cache.update(1000000, market_quintets)

    assert resize == dec(0)
    assert new_quintets == []
    assert len(conversion_cache.caches) == conversion_cache.max_size
    assert i == 9  # 8 should still add, but fails at 9
    assert "0x18" in conversion_cache.caches
    assert "0x19" not in conversion_cache.caches
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        m[0] for m in all_market_quintets
    }
    assert conversion_cache.caches["0x9"].cumulative_size == dec(10)


def test_update_raises_all_market_quintets(conversion_cache):
    # failure cases: gaps in all_market_quintets, mixed neg_risk_market_ids,
    #   empty all_market_quintets, non neg_risk all_market_quintets

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    conversion_cache.update(10, all_market_quintets)

    # incomplete subset
    with pytest.raises(PolyPyException) as record:
        conversion_cache.update(
            None,
            [MarketIdQuintet("0x10", "0x9", "0x005", "Y5", "N5")],
        )
    assert "Missing" in str(record)

    # gaps in all_market_quintets
    with pytest.raises(PolyPyException) as record:
        conversion_cache.update(
            None,
            all_market_quintets + [MarketIdQuintet("0x10", "0x9", "0x005", "Y5", "N5")],
        )
    assert "not consecutive" in str(record)

    # mixed neg_risk_market_ids
    with pytest.raises(PolyPyException) as record:
        conversion_cache.update(
            None,
            all_market_quintets + [MarketIdQuintet("0x4", "0x1", "0x003", "Y4", "N4")],
        )
    assert "do not match" in str(record)

    # empty all_market_quintets
    with pytest.raises(PolyPyException) as record:
        conversion_cache.update(None, [])
    assert "empty" in str(record)

    # non neg_risk all_market_quintets
    with pytest.raises(PolyPyException) as record:
        conversion_cache.update(None, [MarketIdQuintet("0x3", "", "0x002", "Y3", "N3")])
    assert "Non-negative" in str(record)
    with pytest.raises(PolyPyException):
        conversion_cache.update(
            None, [MarketIdQuintet("0x001", "", "0x002", "Y3", "N3")]
        )
    assert "Non-negative" in str(record)
    with pytest.raises(PolyPyException) as record:
        x = all_market_quintets + [MarketIdQuintet("0x001", "", "0x003", "Y3", "N3")]
        conversion_cache.update(
            None,
            x,
        )
    assert "do not match" in str(record)

    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        m[0] for m in all_market_quintets
    }
    assert conversion_cache.caches["0x9"].cumulative_size == dec(10)


def test_pull(conversion_cache, mocker):
    # empty pull
    closed_nrm, yes_tokens = conversion_cache.pull()
    assert closed_nrm == []
    assert yes_tokens == []
    assert len(conversion_cache.caches) == 0

    # pre-load cache
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    conversion_cache.update(10, all_market_quintets)

    # mock
    all_market_quintets.append(MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4"))
    mocker.patch(
        "polypy.manager.cache.get_neg_risk_markets",
        return_value=(
            [False],
            [all_market_quintets],
        ),
    )

    # update existing
    closed_nrm, diff = conversion_cache.pull()
    assert diff[0][0] == dec(10)
    assert diff[0][1] == ["Y4"]
    assert closed_nrm[0][0] == "0x9"
    assert closed_nrm[0][1] == False
    assert len(closed_nrm) == 1
    assert len(diff) == 1

    # no update
    all_market_quintets_2 = [
        MarketIdQuintet("0x11", "0x19", "0x1000", "Y11", "N11"),
        MarketIdQuintet("0x12", "0x19", "0x1001", "Y12", "N12"),
    ]
    conversion_cache.update(1, all_market_quintets_2)
    mocker.patch(
        "polypy.manager.cache.get_neg_risk_markets",
        return_value=(
            [False, False],
            [all_market_quintets, all_market_quintets_2],
        ),
    )
    closed_nrm, diff = conversion_cache.pull()
    assert len(closed_nrm) == 2
    assert len(diff) == 2
    assert diff[0][0] == diff[1][0] == dec(0)
    assert diff[0][1] == diff[1][1] == []
    assert closed_nrm[0][0] == "0x9"
    assert closed_nrm[1][0] == "0x19"
    assert closed_nrm[0][1] == closed_nrm[1][1] == False
    assert len(conversion_cache.caches) == 2
    assert conversion_cache.caches["0x9"].cumulative_size == dec(10)
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        "0x1",
        "0x2",
        "0x3",
        "0x4",
    }
    assert conversion_cache.caches["0x19"].cumulative_size == dec(1)
    assert conversion_cache.caches["0x19"].seen_condition_ids == {"0x11", "0x12"}

    # update existing with no diff and delete closed with diff
    all_market_quintets_2.append(
        MarketIdQuintet("0x13", "0x19", "0x1002", "Y13", "N13")
    )
    mocker.patch(
        "polypy.manager.cache.get_neg_risk_markets",
        return_value=(
            [True, False],
            [all_market_quintets, all_market_quintets_2],
        ),
    )
    closed_nrm, diff = conversion_cache.pull()
    assert len(closed_nrm) == 2
    assert len(diff) == 2
    assert closed_nrm[0] == ("0x9", True)
    assert closed_nrm[1] == ("0x19", False)
    assert diff[0] == (dec(0), [])
    assert diff[1] == (dec(1), ["Y13"])
    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x19"].cumulative_size == dec(1)
    assert conversion_cache.caches["0x19"].seen_condition_ids == {
        "0x11",
        "0x12",
        "0x13",
    }


# noinspection DuplicatedCode
def test_pull_by_id(conversion_cache, mocker):
    # sourcery skip: extract-duplicate-method
    # empty pull
    with pytest.raises(PolyPyException):
        conversion_cache.pull_by_id(None, "0x0")

    # pre-load cache
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    conversion_cache.update(10, all_market_quintets)

    # mock
    all_market_quintets.append(MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4"))
    mocker.patch(
        "polypy.manager.cache.get_neg_risk_markets",
        return_value=(False, all_market_quintets),
    )

    # update existing
    closed_nrm, diff = conversion_cache.pull_by_id(
        condition_id="0x1", neg_risk_market_id=None
    )
    assert diff[0] == dec(10)
    assert diff[1] == ["Y4"]
    assert closed_nrm[0] == "0x9"
    assert closed_nrm[1] == False
    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x9"].cumulative_size == dec(10)
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        "0x1",
        "0x2",
        "0x3",
        "0x4",
    }

    with pytest.raises(PolyPyException) as record:
        conversion_cache.pull_by_id(condition_id="0x1", neg_risk_market_id="some_id")
    assert "not in" in str(record)
    with pytest.raises(PolyPyException) as record:
        conversion_cache.pull_by_id(condition_id="0x111111", neg_risk_market_id="0x9")
    assert "not in" in str(record)

    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x9"].cumulative_size == dec(10)
    assert conversion_cache.caches["0x9"].seen_condition_ids == {
        "0x1",
        "0x2",
        "0x3",
        "0x4",
    }

    # delete
    mocker.patch(
        "polypy.manager.cache.get_neg_risk_markets",
        return_value=(True, all_market_quintets),
    )
    closed_nrm, diff = conversion_cache.pull_by_id(
        condition_id="0x1", neg_risk_market_id=None
    )
    assert diff[0] == dec(0)
    assert diff[1] == []
    assert closed_nrm[0] == "0x9"
    assert closed_nrm[1] == True
    assert len(conversion_cache.caches) == 0

    with pytest.raises(PolyPyException) as record:
        conversion_cache.pull_by_id(condition_id=None, neg_risk_market_id=None)
    assert "None" in str(record)


def test_pull_by_id_raises(conversion_cache):
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]
    all_market_quintets_2 = [
        MarketIdQuintet("0x11", "0x19", "0x1000", "Y11", "N11"),
        MarketIdQuintet("0x12", "0x19", "0x1001", "Y12", "N12"),
    ]

    conversion_cache.update(10, all_market_quintets)
    conversion_cache.update(10, all_market_quintets_2)

    with pytest.raises(PolyPyException) as record:
        conversion_cache.pull_by_id(condition_id="0x11", neg_risk_market_id="0x9")
    assert "cache" in str(record)

    assert len(conversion_cache.caches) == 2
    assert "0x9" in conversion_cache
    assert "0x19" in conversion_cache


def test_locking(conversion_cache):
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1"),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2"),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3"),
    ]

    def f(cache: AugmentedConversionCache):
        for _ in range(100):
            cache.update(dec(1), all_market_quintets)

    threads = [threading.Thread(target=f, args=(conversion_cache,)) for _ in range(100)]
    for thread in threads:
        thread.start()

    time.sleep(4)

    for thread in threads:
        thread.join()

    assert len(conversion_cache.caches) == 1
    assert conversion_cache.caches["0x9"].cumulative_size == 10_000
    assert conversion_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}
