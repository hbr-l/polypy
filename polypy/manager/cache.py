import math
from decimal import Decimal
from threading import RLock
from typing import Iterable, Protocol, TypeVar

import attrs
import msgspec.json

from polypy.constants import ENDPOINT
from polypy.ctf import MarketIdQuintet
from polypy.exceptions import PolyPyException
from polypy.rest.api import (
    get_events_gamma_model,
    get_markets_gamma_model,
    get_neg_risk_market,
)
from polypy.typing import NumericAlias, dec


def _check_conversion_all_quintets(all_quintets: Iterable[MarketIdQuintet]) -> None:
    all_qids = [int(m[2][-2:], 16) for m in all_quintets]

    if not all_qids:
        raise PolyPyException("`all_market_quintets` is empty.")

    # check if all_qids consists of only consecutive numbers
    if not max(all_qids) - min(all_qids) + 1 == len(all_qids) == len(set(all_qids)):
        # todo this check might be too restrictive
        raise PolyPyException(
            f"`all_market_quintets` question_ids are not consecutive: {all_quintets}"
        )


class ConversionCacheProtocol(Protocol):
    def update(
        self, size: NumericAlias | None, all_market_quintets: Iterable[MarketIdQuintet]
    ) -> tuple[Decimal, list[str]]:
        """Update cache with market information of a specific negative risk market
        and return diff containing cumulative converted size and YES token ids (pre- vs. post-update).
        """

    def pull(
        self, endpoint_gamma: str | ENDPOINT
    ) -> tuple[list[str], list[tuple[Decimal, list[str]]]]:
        """Pull information automatically from Gamma API, delete expired caches
        and return closed negative risk market IDs and diffs (cumulative sizes and
        YES token ids per negative risk market)

        Returns
        -------
        closed_neg_risk_market_ids: list[str],
            negative risk market ids of closed events
        diffs: list[tuple[Decimals, list[str]]],
            list of tuples (cumulative converted size, list of YES token ids),
            pre- vs. post-update diff
        """

    def pull_by_condition_id(
        self, size: NumericAlias, condition_id: str
    ) -> tuple[str | None, tuple[Decimal, list[str]]]:
        """Analogous to pull, but for a single negative risk market."""


@attrs.define
class _NegRiskConversionCache:
    neg_risk_market_id: str
    cumulative_size: Decimal
    seen_condition_ids: set[str]


T = TypeVar("T")


def _first_set_element(s: set[T]) -> T:
    for e in s:
        return e


def _get_gamma_events(
    endpoint_gamma: str | ENDPOINT, condition_ids: Iterable[str]
) -> list[dict]:
    gamma_markets: list[dict] = get_markets_gamma_model(
        endpoint_gamma=endpoint_gamma, condition_ids=condition_ids
    )
    gamma_events: list[dict] = get_events_gamma_model(
        endpoint_gamma=endpoint_gamma,
        ids=[m["events"][0]["id"] for m in gamma_markets],
    )
    return gamma_events


def _quintets_from_gamma_event(event: dict) -> list[MarketIdQuintet]:
    quintets = []
    for market in event["markets"]:
        tokens = msgspec.json.decode(market["clobTokenIds"])
        quintets.append(
            MarketIdQuintet(
                condition_id=market["conditionId"],
                neg_risk_market_id=market["negRiskMarketID"],
                question_id=market["questionID"],
                token_id_1=tokens[0],
                token_id_2=tokens[2],
                rest_endpoint=None,
            )
        )
    return quintets


class AugmentedConversionCache(ConversionCacheProtocol):
    def __init__(self, endpoint_gamma: str | ENDPOINT, max_size: int | None = None):
        self.endpoint_gamma = endpoint_gamma
        self.max_size = math.inf if max_size is None else max_size

        self.caches = {}
        self.lock = RLock()

    def _check_max_size(self):
        if len(self.caches) >= self.max_size:
            raise PolyPyException(f"`max_size`={self.max_size} exceeded.")

    def _add_new_cache(
        self,
        negative_risk_id: str,
        size: NumericAlias,
        all_quintets: Iterable[MarketIdQuintet],
    ) -> tuple[Decimal, list[str]]:
        self._check_max_size()

        cache = _NegRiskConversionCache(
            neg_risk_market_id=negative_risk_id,
            cumulative_size=Decimal(0),
            seen_condition_ids=set(),
        )

        condition_ids = {q[0] for q in all_quintets}

        cache.cumulative_size += dec(size)
        cache.seen_condition_ids.update(condition_ids)

        self.caches[negative_risk_id] = cache
        return Decimal(0), []

    def _update_existing_cache(
        self,
        neg_risk_market_id: str,
        size: NumericAlias,
        all_quintets: Iterable[MarketIdQuintet],
    ) -> tuple[Decimal, list[str]]:
        cache = self.caches[neg_risk_market_id]

        # dict[condition_id, token_id_1]
        lookup = {q[0]: q[3] for q in all_quintets}

        # get ids of yet unseen conditions
        diff_condition_ids = lookup.keys() - cache.seen_condition_ids
        token_1_ids = [lookup[i] for i in diff_condition_ids]

        re_size = cache.cumulative_size if token_1_ids else Decimal(0)
        cache.cumulative_size += dec(size)
        cache.seen_condition_ids.update(lookup.keys())

        return re_size, token_1_ids

    def update(
        self, size: NumericAlias | None, all_market_quintets: Iterable[MarketIdQuintet]
    ) -> tuple[Decimal, list[str]]:
        _check_conversion_all_quintets(all_market_quintets)

        neg_risk_market_ids = {q[1] for q in all_market_quintets}

        if len(neg_risk_market_ids) != 1:
            raise PolyPyException(
                "All `all_market_quintets` must contain the same neg_risk_market_id."
            )

        nrm = _first_set_element(neg_risk_market_ids)
        size = Decimal(0) if size is None else size

        with self.lock:
            if nrm in self.caches:
                return self._update_existing_cache(
                    neg_risk_market_id=nrm, size=size, all_quintets=all_market_quintets
                )
            else:
                return self._add_new_cache(
                    negative_risk_id=nrm, size=size, all_quintets=all_market_quintets
                )

    def pull(
        self, endpoint_gamma: str | ENDPOINT
    ) -> tuple[list[str], list[tuple[Decimal, list[str]]]]:
        ret = []
        closed_neg_market_ids = []

        with self.lock:
            condition_ids = [
                _first_set_element(cache.seen_condition_ids)
                for cache in self.caches.values()
            ]

            if not len(condition_ids):
                return [], []

            gamma_events = _get_gamma_events(endpoint_gamma, condition_ids)
            for event in gamma_events:
                quintets = _quintets_from_gamma_event(event)
                size, token_1_ids = self.update(
                    size=Decimal(0), all_market_quintets=quintets
                )
                ret.append((size, token_1_ids))

                if event["closed"]:
                    closed_neg_market_ids.append(event["negRiskMarketID"])

            for closed_nrm in closed_neg_market_ids:
                # if market is closed, no active conversions possible anymore,
                # though claiming positions still possible (this is managed separately in PositionManager)
                del self.caches[closed_nrm]

            return closed_nrm, ret

    def pull_by_condition_id(
        self, size: NumericAlias, condition_id: str
    ) -> tuple[str | None, tuple[Decimal, list[str]]]:
        closed, all_quintets = get_neg_risk_market(
            endpoint_gama=self.endpoint_gamma,
            include_closed=True,
            condition_id=condition_id,
            token_id=None,
            market_slug=None,
        )

        nrm = all_quintets[0][1] if closed else None

        return nrm, self.update(size, all_quintets)
