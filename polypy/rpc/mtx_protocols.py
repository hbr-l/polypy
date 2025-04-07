from enum import StrEnum
from typing import Iterable

import attrs

from polypy.constants import CHAIN_ID, COLLATERAL
from polypy.ctf import MarketIdQuintet
from polypy.exceptions import PolyPyException
from polypy.rpc.encode import (
    encode_convert,
    encode_merge,
    encode_redeem,
    encode_redeem_neg_risk,
    encode_split,
)
from polypy.typing import NumericAlias


def _check_conversion_all_quintets(all_quintets: Iterable[MarketIdQuintet]) -> None:
    all_qids = [int(m[2][-2:], 16) for m in all_quintets]

    if not all_qids:
        raise PolyPyException("`all_market_quintets` is empty.")

    # check if all_qids consists of only consecutive numbers
    if not max(all_qids) - min(all_qids) + 1 == len(all_qids) == len(set(all_qids)):
        # todo this check might be too restrictive
        raise PolyPyException(
            f"`all_market_quintets` question_ids are not consecutive: {all_quintets}"
        )


# noinspection PyPep8Naming
class MTX_TYPE(StrEnum):
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    REDEEM = "REDEEM"
    CONVERT = "CONVERT"


@attrs.define
class SplitTransaction:
    condition_id: str
    amount_usdc: NumericAlias
    neg_risk: bool

    def encode(self, chain_id: CHAIN_ID) -> str:
        # noinspection PyTypeChecker
        return encode_split(
            condition_id=self.condition_id,
            amount_usdc=self.amount_usdc,
            collateral=COLLATERAL[chain_id.name],
        )


@attrs.define
class MergeTransaction:
    condition_id: str
    size: NumericAlias
    neg_risk: bool

    def encode(self, chain_id: CHAIN_ID) -> str:
        # noinspection PyTypeChecker
        return encode_merge(
            condition_id=self.condition_id,
            size=self.size,
            collateral=COLLATERAL[chain_id.name],
        )


@attrs.define
class RedeemTransaction:
    condition_id: str
    size_yes_neg_risk: NumericAlias | None
    size_no_neg_risk: NumericAlias | None
    neg_risk: bool

    def encode(self, chain_id: CHAIN_ID) -> str:
        if self.neg_risk:
            return encode_redeem_neg_risk(
                condition_id=self.condition_id,
                size_yes=self.size_yes_neg_risk,
                size_no=self.size_no_neg_risk,
            )
        else:
            # noinspection PyTypeChecker
            return encode_redeem(
                condition_id=self.condition_id,
                collateral=COLLATERAL[chain_id.name],
            )


@attrs.define
class ConvertTransaction:
    neg_risk_market_id: str
    size: NumericAlias
    question_ids: list[str]

    @property
    def neg_risk(self) -> bool:
        return True

    def encode(self, *_, **__) -> str:
        return encode_convert(
            neg_risk_market_id=self.neg_risk_market_id,
            question_ids=self.question_ids,
            size=self.size,
        )
