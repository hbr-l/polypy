import numpy as np
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON


def main():
    host = "https://clob.polymarket.com"
    key = None
    creds = None
    chain_id = POLYGON
    client = ClobClient(host, key=key, chain_id=chain_id, creds=creds)

    condition_ids = []
    cursor = ""
    rets = 0
    page = 0

    while True:
        print(f"Page nb: {page}")
        data: dict = client.get_simplified_markets(next_cursor=cursor)
        condition_ids.extend(
            [
                d["condition_id"]
                for d in data["data"]
                if d["active"] is True
                and d["closed"] is False
                and d["accepting_orders"] is True
            ]
        )
        rets += len(data["data"])
        page += 1

        new_cursor = data["next_cursor"]
        if new_cursor == "LTE=":
            break

        cursor = new_cursor

    condition_ids = np.array(condition_ids)
    np.savetxt("condition.csv", condition_ids, delimiter=",", fmt="%s")

    print(len(condition_ids), "/", rets)


if __name__ == "__main__":
    main()
