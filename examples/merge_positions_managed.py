"""
Merging by yourself requires sufficient gas to pay!

This approach updates plp.PositionManager automatically.
"""
import time
from decimal import Decimal

import polypy as plp

# MODIFY
condition_id: str = ...  # condition_id/market_id to merge
token_id_yes: str = ...  # token_id/asset_id to merge (YES)
token_id_no: str = ...  # token_id/asset_id to merge (NO)
private_key: str = ...
api_key: str = ...
secret: str = ...
passphrase: str = ...
maker_addr: str = ...  # magic/mail account: displayed on user profile (polymarket page)
signature_type = plp.SIGNATURE_TYPE.POLY_PROXY  # !!! change if not magic/mail account


def main():
    # get balances
    usdc_balance = plp.get_balance(
        plp.ENDPOINT.REST,
        Decimal,
        signature_type,
        private_key,
        api_key,
        secret,
        passphrase,
    )
    w3 = plp.W3POA(plp.ENDPOINT.RPC_POLYGON, private_key, maker_addr)
    pol_balance = plp.get_balance_POL(w3)
    print("start balance USDC: ", usdc_balance)
    print("start balance POL: ", pol_balance)
    print()

    # fetch positions
    positions = plp.get_positions(
        plp.ENDPOINT.DATA,
        maker_addr,
        plp.Position,
        condition_id,
        None,
        None,
        None,
        dtype_size=Decimal,
    )
    # setup and fill position manager with previously fetched positions
    position_manager = plp.PositionManager(
        plp.ENDPOINT.REST, plp.ENDPOINT.GAMMA, usdc_balance
    )
    for position in positions:
        position_manager.track(position)
    print("positions: \n", position_manager.position_dict)
    print()

    rpc = plp.RPCSettings(
        plp.ENDPOINT.RPC_POLYGON,
        None,
        plp.CHAIN_ID.POLYGON,
        private_key,
        maker_addr,
        Decimal(1.4),
        None,
        True,
        None,
        120,
        None,
        None,
        None,
    )
    # merge AND update plp.PositionManager
    relayer_response, tx_hash, tx_receipt = position_manager.merge_positions(
        plp.MarketIdTriplet(condition_id, token_id_yes, token_id_no),
        min(pos.size for pos in positions),
        rpc,
    )
    print("relayer_response: ", relayer_response)
    print("tx_hash: ", tx_hash)
    print("tx_receipt: ", tx_receipt)
    print()

    time.sleep(
        20
    )  # give some time until polymarket has processed updates on their side...

    usdc_balance = position_manager.balance
    pol_balance = plp.get_balance_POL(w3)
    positions = position_manager.position_dict
    print("new balance USDC: ", usdc_balance)
    print("new balance POL: ", pol_balance)
    print("new positions: \n", positions)


if __name__ == "__main__":
    main()
