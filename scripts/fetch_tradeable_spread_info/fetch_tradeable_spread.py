import concurrent.futures
import datetime
import threading
import time
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderSummary
from py_clob_client.constants import POLYGON
from py_clob_client.exceptions import PolyApiException


def get_yes_no_token(market):
    if market["tokens"][0]["outcome"] == "Yes":
        # token_id, out_come, price, winner, tags
        yes_token: dict = market["tokens"][0]
        no_token: dict = market["tokens"][1]
    else:
        yes_token: dict = market["tokens"][1]
        no_token: dict = market["tokens"][0]

    return yes_token, no_token


def get_order_book_info(yes_token, client):
    order_book = client.get_order_book(yes_token["token_id"])

    try:
        best_bid = order_book.bids[-1]
        best_ask = order_book.asks[-1]
        spread = Decimal(best_ask.price) - Decimal(best_bid.price)
    except IndexError:
        best_bid = OrderSummary("-1", "-1")
        best_ask = OrderSummary("-1", "-1")
        spread = Decimal(-1)

    return spread, best_bid, best_ask


def compile_info(
    c_id, yes_token, no_token, spread_yes, best_bid_yes, best_ask_yes, market
):
    return [
        c_id,
        yes_token["token_id"],
        no_token["token_id"],
        spread_yes,
        best_bid_yes.price,
        best_bid_yes.size,
        best_ask_yes.price,
        best_ask_yes.size,
        market["minimum_order_size"],
        market["minimum_tick_size"],
        market["market_slug"],
        market["end_date_iso"],
        market["maker_base_fee"],
        market["taker_base_fee"],
        market["neg_risk"],
    ]


def retry_decorator(times, exceptions, delay):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    warnings.warn(
                        f"Retrying {attempt}/{times} in {delay} s. "
                        f"Exception: {str(e)}. args: {args}, kwargs: {kwargs}."
                    )
                    time.sleep(delay)
            return func(*args, **kwargs)

        return newfn

    return decorator


@retry_decorator(times=4, exceptions=PolyApiException, delay=60)
def get_info(condition_idx, client, throttle) -> list:
    market = client.get_market(condition_idx)

    if market["enable_order_book"] is False:
        print(f"Order book not enabled for {condition_idx}.")
        return []

    if datetime.datetime.fromisoformat(market["end_date_iso"]) <= datetime.datetime.now(
        datetime.timezone.utc
    ):
        print(
            f"Market expired: {datetime.datetime.fromisoformat(market['end_date_iso'])} "
            f"({datetime.datetime.now(datetime.timezone.utc)}) - {condition_idx}."
        )
        return []

    yes_token, no_token = get_yes_no_token(market)
    time.sleep(throttle)
    spread_yes, best_bid_yes, best_ask_yes = get_order_book_info(yes_token, client)

    return compile_info(
        condition_idx,
        yes_token,
        no_token,
        spread_yes,
        best_bid_yes,
        best_ask_yes,
        market,
    )


def proc_chunk(
    condition_ids: list | np.ndarray, client: ClobClient, intra_throttle, inter_throttle
) -> list[list]:
    data = []

    for i, c_id in enumerate(condition_ids):
        print(f"Fetching: ({threading.get_ident()}) {i+1}/{len(condition_ids)}")

        try:
            info = get_info(c_id, client, intra_throttle)
            data.append(info)
        except Exception as e:
            warnings.warn(f"{str(e)}. Condition_id {c_id}.")

        time.sleep(inter_throttle)

    return data


def finalize_save(results: list[list]):
    results = pd.DataFrame(
        results,
        columns=[
            "condition_id",
            "token_id_yes",
            "token_id_no",
            "spread_yes",
            "best_bid_price_yes",
            "best_bid_size_yes",
            "best_ask_price_yes",
            "best_ask_size_yes",
            "minimum_order_size",
            "minimum_tick_size",
            "market_slug",
            "end_date_iso",
            "maker_base_fee",
            "taker_base_fee",
            "neg_risk",
        ],
    )

    results = results.sort_values(
        by=["spread_yes"], key=lambda col: col.astype(float), ascending=False
    )
    results.to_csv("market_info.csv", index=False)


def prepare_chunks(condition_ids, n_threads):
    chunks = np.array_split(condition_ids, min(n_threads, len(condition_ids)))
    print(
        f"{len(chunks)} chunks with avg. {np.average([len(c) for c in chunks])} items."
    )
    return chunks


def main():
    host = "https://clob.polymarket.com"
    chain_id = POLYGON
    client = ClobClient(host, key=None, chain_id=chain_id, creds=None)
    n_threads = 500
    intra_throttle = 1
    inter_throttle = 2
    mod5_throttle = 5
    mod10_throttle = 25
    mod100_throttle = 70

    condition_ids = np.loadtxt("condition.csv", delimiter=",", dtype=str)
    chunks = prepare_chunks(condition_ids, n_threads)

    with warnings.catch_warnings():
        warnings.simplefilter("always")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        proc_chunk, chunk, client, intra_throttle, inter_throttle
                    )
                )
                time.sleep(0.1)

                if (i % 5) == 0:
                    time.sleep(mod5_throttle)
                elif (i % 10) == 0:
                    time.sleep(mod10_throttle)
                elif (i % 100) == 0:
                    time.sleep(mod100_throttle)

            concurrent.futures.wait(futures)

            results = []
            for f in futures:
                results.extend(f.result())

    print(f"Fetched: {len(results)}/{len(condition_ids)}")

    finalize_save(results)


if __name__ == "__main__":
    main()
