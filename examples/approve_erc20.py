import time

from polypy.constants import CHAIN_ID, ENDPOINT
from polypy.rpc import (
    W3POA,
    auto_approve_USDC,
    get_allowance_USDC,
    is_sufficient_approval_erc20,
)
from polypy.typing import NumericAlias


def main(private_key: str, amount: NumericAlias):
    maker_addr = None

    w3 = W3POA(ENDPOINT.RPC_POLYGON, private_key, maker_addr)

    curr = get_allowance_USDC(w3, CHAIN_ID.POLYGON)
    print("Current allowance: ", curr)

    sufficient = is_sufficient_approval_erc20(w3, amount, CHAIN_ID.POLYGON)
    print("Is sufficient: ", sufficient)

    tx_hash, receipt = auto_approve_USDC(
        w3,
        amount,
        private_key,
        CHAIN_ID.POLYGON,
        1.2,
        None,
        120,
    )
    print("Approval data: ", tx_hash, receipt)

    if tx_hash is not None:
        time.sleep(10)

    curr = get_allowance_USDC(w3, CHAIN_ID.POLYGON)
    print("Current allowance: ", curr)


if __name__ == "__main__":
    pk = input("private_key: ")
    am = float(input("Amount: "))
    main(pk, am)
