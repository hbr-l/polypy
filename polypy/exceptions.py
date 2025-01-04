class PolyPyException(Exception):
    """Base exception for polypy."""


class RewardException(PolyPyException):
    """Exceptions related to rewards."""


class OrderBookException(PolyPyException):
    """Exceptions related to order book retrieval, storage and update."""


class EventTypeException(PolyPyException):
    """Exception related to event_type field of websocket messages."""


class OrderCreationException(PolyPyException):
    """Exception related to order creation."""


class OrderUpdateException(PolyPyException):
    """
    Exception related to updating order.
    Order is created and might or might not be posted/ submitted to exchange already.
    """
