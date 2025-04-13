from enum import IntEnum, StrEnum

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ZERO_HASH = "0x0000000000000000000000000000000000000000000000000000000000000000"

SIG_DIGITS_SIZE = 2

USDC = "usdc"

ERC20_WEI_UNIT = "mwei"
POL_WEI_UNIT = "ether"
APPROVAL_MARGIN_WEI = 1
# if you absolutely have to modify APPROVAL_MARGIN_WEI dynamically:
#   from polypy.rpc import encode
#   encode.APPROVAL_MARGIN_WEI = 0


# noinspection PyPep8Naming
class CHAIN_ID(IntEnum):
    POLYGON = 137
    AMOY = 80002


# noinspection PyPep8Naming
class CONDITIONAL(StrEnum):
    POLYGON = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    AMOY = "0x69308FB512518e39F9b16112fA8d994F4e2Bf8bB"


class COLLATERAL(StrEnum):
    POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    AMOY = "0x9c4e1703476e875070ee25b56a58b008cfb8fa78"


# noinspection PyPep8Naming
class EXCHANGE(StrEnum):
    POLYGON = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    AMOY = "0xdFE02Eb6733538f8Ea35D585af8DE5958AD99E40"
    POLYGON_NEG_RISK = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    AMOY_NEG_RISK = "0xC5d563A36AE78145C45a50134d48A1215220f80a"


# noinspection PyPep8Naming
class NEGATIVE_RISK_ADAPTER(StrEnum):
    POLYGON = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    AMOY = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"


class ENDPOINT(StrEnum):
    REST = "https://clob.polymarket.com"
    WS = "wss://ws-subscriptions-clob.polymarket.com/ws"
    DATA = "https://data-api.polymarket.com"
    RELAYER = "https://relayer-v2.polymarket.com"
    RPC_POLYGON = "https://polygon-rpc.com"
    GAMMA = "https://gamma-api.polymarket.com"


RELAY_HUB = "0xD216153c06E857cD7f72665E0aF1d7D82172F494"
PROXY_WALLET_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
