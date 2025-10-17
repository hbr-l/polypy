"""
Merging by yourself requires sufficient gas to pay!

Functional approach does not update plp.PositionManager!
"""
import time
from decimal import Decimal

import polypy as plp

# MODIFY
condition_id: str = ...  # condition_id/market_id to merge
token_id: str = (
    ...
)  # token_id/asset_id to merge, can be YES or NO (only used to determine neg_risk)
private_key: str = ...
api_key: str = ...
secret: str = ...
passphrase: str = ...
maker_addr: str = ...  # magic/mail account: displayed on user profile (polymarket page)
signature_type = plp.SIGNATURE_TYPE.POLY_PROXY  # !!! change if not magic/mail account


def main():
    # get balance
    w3 = plp.W3POA(plp.ENDPOINT.RPC_POLYGON, private_key, maker_addr)
    usdc_balance = plp.get_balance(
        plp.ENDPOINT.REST,
        Decimal,
        signature_type,
        private_key,
        api_key,
        secret,
        passphrase,
    )
    pol_balance = plp.get_balance_POL(w3)
    print("start balance USDC: ", usdc_balance)
    print("start balance POL: ", pol_balance)
    print()

    positions = plp.get_positions(
        plp.ENDPOINT.DATA, maker_addr, plp.Position, condition_id, None, None, None
    )
    neg_risk = plp.get_neg_risk(plp.ENDPOINT.REST, token_id)
    print("positions: ", positions)
    print("neg_risk: ", neg_risk)
    print()

    # functional approach: THIS WILL NOT UPDATE plp.PositionManager
    relayer_response, tx_hash, tx_receipt = plp.merge_positions(
        w3,
        condition_id,
        min(pos.size for pos in positions),
        neg_risk,
        plp.CHAIN_ID.POLYGON,
        private_key,
        maker_addr,
        1.4,
        None,
        True,
        None,
        None,
        None,
        None,
        None,
        120,
    )
    print("relayer_response: ", relayer_response)
    print("tx_hash: ", tx_hash)
    print("tx_receipt: ", tx_receipt)
    print()

    time.sleep(
        20
    )  # give some time until polymarket has processed updates on their side...

    usdc_balance = plp.get_balance(
        plp.ENDPOINT.REST,
        Decimal,
        signature_type,
        private_key,
        api_key,
        secret,
        passphrase,
    )
    pol_balance = plp.get_balance_POL(w3)
    positions = plp.get_positions(
        plp.ENDPOINT.DATA, maker_addr, plp.Position, condition_id, None, None, None
    )
    print("new balance USDC: ", usdc_balance)
    print("new balance POL: ", pol_balance)
    print("new positions: ", positions)


if __name__ == "__main__":
    main()
