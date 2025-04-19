from enum import StrEnum

import attrs

from polypy.constants import CHAIN_ID, COLLATERAL
from polypy.rpc.encode import (
    encode_convert,
    encode_merge,
    encode_redeem,
    encode_redeem_neg_risk,
    encode_split,
)
from polypy.typing import NumericAlias


# noinspection PyPep8Naming
class RPC_TYPE(StrEnum):
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
