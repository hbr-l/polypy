from polypy.ctf import MarketIdQuintet, MarketIdTriplet


# noinspection DuplicatedCode
def test_quintet():
    m = MarketIdQuintet("0", "1", "2", "3", "4")

    for i in range(5):
        assert m[i] == str(i)
    assert m.condition_id == "0"
    assert m.neg_risk_market_id == "1"
    assert m.question_id == "2"
    assert m.token_id_1 == "3"
    assert m.token_id_2 == "4"


# noinspection DuplicatedCode
def test_quintet_create():
    m = MarketIdQuintet.create("0", "1", "2", "3", "4", None)

    for i in range(5):
        assert m[i] == str(i)
    assert m.condition_id == "0"
    assert m.neg_risk_market_id == "1"
    assert m.question_id == "2"
    assert m.token_id_1 == "3"
    assert m.token_id_2 == "4"


# noinspection DuplicatedCode
def test_triplet():
    m = MarketIdTriplet("0", "1", "2")

    for i in range(3):
        assert m[i] == str(i)
    assert m.condition_id == "0"
    assert m.token_id_1 == "1"
    assert m.token_id_2 == "2"


# noinspection DuplicatedCode
def test_triplet_create():
    m = MarketIdTriplet.create("0", "1", "2", None)

    for i in range(3):
        assert m[i] == str(i)
    assert m.condition_id == "0"
    assert m.token_id_1 == "1"
    assert m.token_id_2 == "2"
