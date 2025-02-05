from enum import StrEnum


# noinspection PyPep8Naming
class TRADE_STATUS(StrEnum):
    MATCHED = "MATCHED"
    MINED = "MINED"
    CONFIRMED = "CONFIRMED"
    RETRYING = "RETRYING"
    FAILED = "FAILED"


# noinspection PyPep8Naming
class TRADER_SIDE(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"
