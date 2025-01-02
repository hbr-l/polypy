import contextlib
import datetime
import json
import threading
import time
import warnings
from collections import deque
from queue import Empty, Queue

import dash
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from polypy.orderbook import OrderBook, message_to_orderbook

data_queue = Queue()
msg_deque = deque(maxlen=5)
msg_queue = Queue()

app = dash.Dash(__name__)
# noinspection PyTypeChecker
app.layout = html.Div(
    style={
        "backgroundColor": "#1e1e1e",  # Dark background
        "color": "#f5f5f5",  # Light text
        "fontFamily": "Arial, sans-serif",
        "padding": "0",  # No padding
        "margin": "0",  # No margin
        "boxSizing": "border-box",  # Ensure no extra spacing
        "height": "100vh",  # Full height of the viewport
        "outline": "0",
    },
    children=[
        html.H1(
            "Live Order Book",
            style={
                "textAlign": "center",
                "marginBottom": "20px",
                "color": "#ffffff",
                "padding": "0",  # No padding
                "margin": "0",  # No margin
            },
        ),
        html.Div(
            [
                dcc.Graph(
                    id="orderbook-plot",
                    figure=go.Figure(
                        data=[
                            go.Bar(
                                # todo magic numbers
                                x=np.round(np.linspace(1, 0, 1001), 3),
                                y=np.zeros(1001),
                                marker_color="#21bf73",  # Modern green
                                width=0.001,
                                name="bid",
                            ),
                            go.Bar(
                                x=np.round(np.linspace(0, 1, 1001), 3),
                                y=np.zeros(1001),
                                marker_color="#e94e77",  # Modern red
                                width=0.001,
                                name="ask",
                            ),
                        ],
                        layout={"xaxis": {"range": [0, 1]}},
                    ),
                    style={
                        "height": "60vh",
                        "width": "80%",
                        "marginBottom": "20px",
                        "marginTop": "10px",
                        "backgroundColor": "#2e2e2e",  # Match dark mode
                        "outline": "0"
                        # "borderRadius": "10px",
                        # "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                    },
                ),
                dcc.Textarea(
                    id="out",
                    value="",
                    style={
                        "width": "80%",
                        "height": "20vh",
                        "margin": "0 auto",
                        "display": "block",
                        "backgroundColor": "#2e2e2e",  # Match dark mode
                        "color": "#f5f5f5",
                        "border": "none",  # Removed border
                        # "borderRadius": "10px",
                        "padding": "10px",
                        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "padding": "0",  # No padding
                "margin": "0",  # No margin
                "outline": "0",
            },
        ),
        dcc.Interval(
            id="interval-component",
            interval=100,  # Update every 100 milliseconds
            n_intervals=0,
        ),
    ],
)


@app.callback(
    Output("orderbook-plot", "figure"),
    [Input("interval-component", "n_intervals"), Input("orderbook-plot", "figure")],
)
def update_orderbook(_, figure):
    with contextlib.suppress(Empty):
        bid_q, ask_q = data_queue.get_nowait()

        figure["data"][0]["y"] = np.cumsum(bid_q)
        figure["data"][1]["y"] = np.cumsum(ask_q)

    return figure


@app.callback(
    Output("out", "value"),
    [Input("interval-component", "n_intervals"), Input("out", "value")],
)
def update_output(_, value):
    with contextlib.suppress(Empty):
        msg_queue.get_nowait()

        value = "\n\n".join(str(x) for x in list(msg_deque)[::-1])

    return value


def write_csv(msg, book, fn, fn_txt):
    # append timestamp and message type
    if msg["event_type"] == "tick_size_change":
        return  # write nothing

    bids = book.bids[1].tolist()
    asks = book.asks[1].tolist()
    event = [msg["event_type"]]
    timestamp = [msg["timestamp"]]

    row = bids + asks + event + timestamp
    assert len(row) == 2004

    str_row = ",".join(str(x) for x in row)
    str_row = f"{str_row}\n"

    with open(fn, "a") as f:
        f.write(str_row)

    with open(fn_txt, "a") as f:
        f.write(f"{str(msg)}\n")


def ws_listener(url, params, book, fn, fn_txt):
    # logger = logging.getLogger("websockets")
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(logging.StreamHandler())

    while True:
        with connect(url) as websocket:
            websocket.send(json.dumps(params))
            while True:
                try:
                    message = websocket.recv(5)
                    message = json.loads(message)
                    for msg in message:
                        if msg["event_type"] in {
                            "tick_size_change",
                            "price_change",
                            "book",
                        }:
                            book, _ = message_to_orderbook(msg, book)
                        else:
                            print(msg)
                        write_csv(msg, book, fn, fn_txt)

                    data_queue.put_nowait([book.bids[1], book.asks[1]])
                    msg_deque.append(message)
                    msg_queue.put_nowait(1)
                except TimeoutError:
                    websocket.ping()
                except ConnectionClosedError as e:
                    warnings.warn(
                        f"{datetime.datetime.now()}: Re-connecting. Traceback: {str(e)}"
                    )
                    break


def main():
    orderbook = OrderBook(
        "101669189743438912873361127612589311253202068943959811456820079057046819967115",
        0.001,
    )
    fn = "messages_4_fed_cuts.csv"

    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    params = {
        "auth": {},
        "markets": [],
        "assets_ids": [orderbook.token_id],
        "type": "market",
    }

    fn_txt = f"_{fn.split('.')[0]}.txt"

    print(f"Writting to: {fn} and {fn_txt}")

    thread = threading.Thread(
        target=ws_listener,
        args=(url, params, orderbook, fn, fn_txt),
    )
    thread.start()
    time.sleep(1)

    app.run_server(debug=True)


if __name__ == "__main__":
    main()
