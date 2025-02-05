class PolyPyException(Exception):
    """Base exception for polypy."""


class RewardException(PolyPyException):
    """Exceptions related to rewards."""


class OrderBookException(PolyPyException):
    """Exceptions related to order book retrieval, storage and update."""


class StreamException(PolyPyException):
    """Base exception for websocket streams."""


class SubscriptionException(StreamException):
    """Exception related to websocket subscription"""


class EventTypeException(PolyPyException):
    """Exception related to event_type field of websocket messages."""


class OrderException(PolyPyException):
    """Order base exception."""


class OrderCreationException(OrderException):
    """Exception related to order creation."""


class OrderPlacementException(OrderException):
    """Base exception related to order placement."""


class OrderPlacementFailure(OrderPlacementException):
    """Exception placing the order: placement was not successful either
    due to server-side or client-side error."""


class OrderPlacementDelayed(OrderPlacementException):
    """Exception during order placement: order marketable but subject to matching delay.
    Not considered as failure."""


class OrderPlacementMarketNotReady(OrderPlacementException):
    """Exception during order placement: system not accepting orders for market yet.
    Not considered as failure."""


class OrderPlacementUnmatched(OrderPlacementException):
    """Exception during order placement: order marketable, but failure delaying, placement not successful"""


class OrderUpdateException(OrderException):
    """
    Exception related to updating order.
    Order is created and might or might not be posted/ submitted to exchange already.
    """


class PositionException(PolyPyException):
    """Base exception related to Positions."""


class PositionNegativeException(PositionException):
    """Negative position balance not allowed for this particular position."""


class ManagerException(PolyPyException):
    """Base exception for managers, e.g. Order Manager or Position Manager."""


class ManagerInvalidException(ManagerException):
    """Manager invalidated."""


class OrderTrackingException(ManagerException):
    """Exception related to tracking the order by, e.g. an Order Manager."""


class PositionTransactionException(ManagerException):
    """Exception during transacting position."""


class PositionTrackingException(ManagerException):
    """Exception related to tracking the position by, e.g. a Position Manager."""
