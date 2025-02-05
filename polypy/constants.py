from enum import IntEnum, StrEnum

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

SIG_DIGITS_SIZE = 2


# noinspection PyPep8Naming
class CHAIN_ID(IntEnum):
    POLYGON = 137
    AMOY = 80002


# noinspection PyPep8Naming
class CONDITIONAL_TOKENS(StrEnum):
    POLYGON = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    AMOY = "0x69308FB512518e39F9b16112fA8d994F4e2Bf8bB"


class COLLATERAL(StrEnum):
    POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    AMOY = "0x9c4e1703476e875070ee25b56a58b008cfb8fa78"


# noinspection PyPep8Naming
class EXCHANGE_ADDRESS(StrEnum):
    POLYGON = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    AMOY = "0xdFE02Eb6733538f8Ea35D585af8DE5958AD99E40"
    POLYGON_NEG_RISK = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    AMOY_NEG_RISK = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"


class ENDPOINT(StrEnum):
    REST = "https://clob.polymarket.com"
    WS = "wss://ws-subscriptions-clob.polymarket.com/ws"
    DATA = "https://data-api.polymarket.com"
