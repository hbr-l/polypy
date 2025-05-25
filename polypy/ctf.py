"""Conditional Token Framework (ctf)"""


from collections import namedtuple
from functools import lru_cache
from typing import Any, Self, TypeVar

from polypy.constants import ENDPOINT
from polypy.exceptions import PolyPyException
from polypy.rest.api import get_market
from polypy.structs import MarketInfo

T = TypeVar("T")


def _optional_equiv(x: Any | None, cmp: T) -> T:
    if x is not None and x != cmp:
        raise PolyPyException(f"Specified x={x} does not match fetched cmp={cmp}.")
    return cmp


@lru_cache(maxsize=128)
def _get_market_ids(endpoint: str | ENDPOINT, condition_id: str):
    market = get_market(endpoint, condition_id)
    return (
        market.neg_risk_market_id,
        market.question_id,
        market.tokens[0].token_id,
        market.tokens[1].token_id,
    )


_MarketIdQuintetBase = namedtuple(
    "MarketIdQuintet",
    [
        "condition_id",  # 0
        "neg_risk_market_id",  # 1
        "question_id",  # 2
        "token_id_1",  # 3: YES
        "token_id_2",  # 4: NO
    ],
)


# noinspection PyArgumentList
class MarketIdQuintet(_MarketIdQuintetBase):
    __slots__ = ()

    @classmethod
    def create(
        cls,
        condition_id: str,
        neg_risk_market_id: str | None,
        question_id: str | None,
        token_id_1: str | None,
        token_id_2: str | None,
        rest_endpoint: str | ENDPOINT | None,
    ) -> Self:
        if any(
            x is None for x in [neg_risk_market_id, question_id, token_id_1, token_id_2]
        ):
            nrm_id, q_id, t_id_1, t_id_2 = _get_market_ids(rest_endpoint, condition_id)

            token_id_1 = _optional_equiv(token_id_1, t_id_1)
            token_id_2 = _optional_equiv(token_id_2, t_id_2)
            question_id = _optional_equiv(question_id, q_id)
            neg_risk_market_id = _optional_equiv(neg_risk_market_id, nrm_id)
        return cls(
            condition_id, neg_risk_market_id, question_id, token_id_1, token_id_2
        )

    @classmethod
    def from_market_info(cls, market_info: MarketInfo) -> Self:
        return cls(
            market_info.condition_id,
            market_info.neg_risk_market_id,
            market_info.question_id,
            market_info.tokens[0].token_id,
            market_info.tokens[1].token_id,
        )

    @classmethod
    def from_tuple(
        cls,
        condition_id: str,
        neg_risk_market_id: str,
        question_id: str,
        token_id_1: str,
        token_id_2: str,
    ) -> Self:
        return cls(
            condition_id=condition_id,
            neg_risk_market_id=neg_risk_market_id,
            question_id=question_id,
            token_id_1=token_id_1,
            token_id_2=token_id_2,
        )


_MarketIdTripletBase = namedtuple(
    "MarketIdTriplet", ["condition_id", "token_id_1", "token_id_2"]
)


# noinspection PyArgumentList
class MarketIdTriplet(_MarketIdTripletBase):
    __slots__ = ()

    @classmethod
    def create(
        cls,
        condition_id: str,
        token_id_1: str | None,
        token_id_2: str | None,
        rest_endpoint: str | ENDPOINT | None,
    ) -> Self:
        if token_id_1 is None or token_id_2 is None:
            _, _, t_id_1, t_id_2 = _get_market_ids(rest_endpoint, condition_id)

            token_id_1 = _optional_equiv(token_id_1, t_id_1)
            token_id_2 = _optional_equiv(token_id_2, t_id_2)

        return cls(condition_id, token_id_1, token_id_2)

    @classmethod
    def from_market_info(cls, market_info: MarketInfo) -> Self:
        return cls(
            market_info.condition_id,
            market_info.tokens[0].token_id,
            market_info.tokens[1].token_id,
        )

    @classmethod
    def from_market_quintet(cls, market_quintet: MarketIdQuintet) -> Self:
        return cls(
            condition_id=market_quintet[0],
            token_id_1=market_quintet[3],
            token_id_2=market_quintet[4],
        )

    @classmethod
    def from_tuple(cls, condition_id: str, token_id_1: str, token_id_2: str) -> Self:
        return cls(
            condition_id=condition_id,
            token_id_1=token_id_1,
            token_id_2=token_id_2,
        )
