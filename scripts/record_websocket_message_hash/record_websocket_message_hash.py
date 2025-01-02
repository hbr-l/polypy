import json

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from websockets.sync.client import connect


def main():
    params = {
        "auth": {},
        "markets": [],
        "assets_ids": [
            "72936048731589292555781174533757608024096898681344338816447372274344589246891"
        ],
        "type": "market",
    }

    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    host = "https://clob.polymarket.com"

    client = ClobClient(host, key=None, chain_id=POLYGON, creds=None)

    ws_msgs_price = []
    rest_msgs_book = []

    break_state = False
    counter = 0
    max_consecutive_prices = 2

    with connect(url) as websocket:
        websocket.send(json.dumps(params))
        while True:
            try:
                message = websocket.recv(30)
                message = json.loads(message)
                for msg in message:
                    if msg["event_type"] == "book":
                        ws_message_book = msg
                        counter = 0
                        ws_msgs_price = []
                        rest_msgs_book = []
                    elif msg["event_type"] == "price_change":
                        rest_msgs_book.append(
                            client.get_order_book(params["assets_ids"][0])
                        )
                        ws_msgs_price.append(msg)
                        counter += 1

                        if counter == max_consecutive_prices:
                            break_state = True
                            break
                    else:
                        print(msg)
            except TimeoutError:
                websocket.ping()

            if break_state:
                break

    for i in range(len(ws_msgs_price)):
        if ws_msgs_price[i]["hash"] != rest_msgs_book[i].hash:
            raise RuntimeError("Hashes do not conform.")

    with open("messages_hash.txt", "w") as f:
        f.write("Initial book message:\n")
        f.write(f"{json.dumps(ws_message_book)}\n")

        for i in range(len(ws_msgs_price)):
            f.write("\n")
            f.write(f"Consecutive price change & REST book message: {i}\n")
            f.write(f"{json.dumps(ws_msgs_price[i])}\n")
            f.write(f"{rest_msgs_book[i].json}\n")
    print()
    print("==================================================")
    print()
    print(ws_message_book)
    print()
    print(ws_msgs_price)
    print()
    print(rest_msgs_book)


if __name__ == "__main__":
    main()
