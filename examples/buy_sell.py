import time
from decimal import Decimal

import polypy as plp
from polypy.structs import OrderWSInfo, TradeWSInfo

# MODIFY
token_id: str = ...  # token_id/asset_id to trade
private_key: str = ...
api_key: str = ...
secret: str = ...
passphrase: str = ...
maker_addr: str = ...  # magic/mail account: displayed on user profile (polymarket page)
signature_type = plp.SIGNATURE_TYPE.POLY_PROXY  # !!! change if not magic/mail account


def print_user_stream_msg(_, msg: TradeWSInfo | OrderWSInfo) -> None:
    print(msg)


def main():
    # get balance
    usdc_balance = plp.get_balance(
        plp.ENDPOINT.REST,
        Decimal,
        signature_type,
        private_key,
        api_key,
        secret,
        passphrase,
    )
    print("start balance: ", usdc_balance)
    print()

    # set up order_manager and position_manager
    order_manager = plp.OrderManager(
        plp.ENDPOINT.REST,
        private_key,
        api_key,
        secret,
        passphrase,
        maker_addr,
        signature_type,
        plp.CHAIN_ID.POLYGON,
    )
    position_manager = plp.PositionManager(
        plp.ENDPOINT.REST, plp.ENDPOINT.GAMMA, usdc_balance
    )

    # setup order book
    tick_size = plp.get_tick_size(plp.ENDPOINT.REST, token_id)
    order_book = plp.OrderBook(token_id, tick_size, plp.zeros_dec)

    # market stream will keep local order book up-to-date
    # user stream will update all changes to positions (in position_manager)
    #   and active orders (in order_manager) which are due to any trade activity
    market_stream = plp.MarketStream(
        plp.ENDPOINT.WS, order_book, None, plp.ENDPOINT.REST
    )
    user_stream = plp.UserStream(
        plp.ENDPOINT.WS,
        (order_manager, position_manager),
        [],  # get updates for any market
        api_key,
        secret,
        passphrase,
        callback_msg=print_user_stream_msg,  # print stream messages
    )

    market_stream.start()
    user_stream.start()
    time.sleep(1)  # give some time for streams to properly connect

    # market buy for 1 USDC
    frozen_order, order_response = order_manager.market_order(
        Decimal(1), token_id, plp.SIDE.BUY, order_book.tick_size, order_book, None
    )
    print("frozen_order (buy): ", frozen_order)
    print("order_response (buy): ", order_response)
    print("order_status (buy): ", order_manager.get_by_id(frozen_order.id).status)
    print()

    # give some time for the trade to settle
    time.sleep(10)
    print("new balance (after buy): ", position_manager.balance)
    print("new position (buy): ", position_manager.get_by_id(token_id))
    print()

    # limit sell previously bought position
    frozen_order, order_response = order_manager.limit_order(
        order_book.best_bid_price,
        position_manager.get_by_id(token_id).size,  # get previous market order size
        token_id,
        plp.SIDE.SELL,
        order_book.tick_size,
        plp.TIME_IN_FORCE.GTC,
        None,
    )
    print("frozen_order (sell): ", frozen_order)
    print("order_response (sell): ", order_response)
    print("order_status (sell): ", order_manager.get_by_id(frozen_order.id).status)
    print()
    time.sleep(0.5)  # give some time such that user_stream can update position_manager
    print("new balance (after sell): ", position_manager.balance)
    print("new position (sell): ", position_manager.get_by_id(token_id))

    # close all streams properly
    market_stream.stop(True, 3)
    user_stream.stop(True, 3)


if __name__ == "__main__":
    main()
