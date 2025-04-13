import math
from decimal import Decimal
from threading import RLock
from typing import Iterable, KeysView, Protocol, TypeVar

import attrs

from polypy.constants import ENDPOINT
from polypy.ctf import MarketIdQuintet
from polypy.exceptions import PolyPyException
from polypy.rest.api import get_neg_risk_markets
from polypy.typing import NumericAlias, dec, is_all_none


class ConversionCacheProtocol(Protocol):
    def __contains__(self, neg_risk_market_id: str) -> bool:
        ...

    def update(
        self, size: NumericAlias | None, all_market_quintets: Iterable[MarketIdQuintet]
    ) -> tuple[Decimal, list[str]]:
        """Update cache with market information of a specific negative risk market
        and return cumulative converted size and YES token ids which were missed so far.

        This is especially useful in case an additional outcome/condition is added to a negative risk market.
        `update(size, all_market_quintets)` takes the size and all market info from the current conversionPositions
        action and returns which YES tokens have been missed so far and the cumulative size, which they
        missed. In doing so, returned cumulative size has to be added to the returned YES tokens on top of
        the `size` input argument (the `size`input argument is passed to the update method to increment the
        cumulative size afterward).

        Parameters
        ----------
        size: NumericAlias | None,
            None equals 0
        all_market_quintets: Iterable[MarketIdQuintet]
            market info including (!) closed markets

        Returns
        -------
        cumulative_size: Decimal,
            cumulative size which needs to be added on top to returned `yes_token_ids`
        yes_token_ids: list[str],
            YES token ids which have been missed so far e.g., new outcome/condition has been added

        Notes
        -----
        There are some basic checks on neg_risk_market_id and question_id w.r.t. MarketIdQuintet, though
        condition_ids and token_ids are not checked! Though, this shouldn't be a problem
        if the correct Polymarket data is fetched accordingly (else this would be Polymarket server-side
        problem...).
        """

    def pull(self) -> tuple[list[tuple[str, bool]], list[tuple[Decimal, list[str]]]]:
        """Pull information automatically from Gamma API, delete expired caches
        and return closed negative risk market IDs and diffs (cumulative sizes and
        YES token ids per negative risk market)

        Returns
        -------
        closed_neg_risk_market_ids: list[tuple[str, bool]],
            tuple of negative risk market ids and event closed status (True if closed)
        diffs: list[tuple[Decimals, list[str]]],
            list of tuples (cumulative converted size, list of YES token ids),
            pre- vs. post-update diff
        """

    def pull_by_id(
        self, condition_id: str | None, neg_risk_market_id: str | None
    ) -> tuple[tuple[str, bool], tuple[Decimal, list[str]]]:
        """Analogous to pull, but for a single negative risk market.
        Raises PolyPyException if market not in cache."""


@attrs.define
class _NegRiskConversionCache:
    neg_risk_market_id: str
    cumulative_size: Decimal
    seen_condition_ids: set[str]


T = TypeVar("T")


def _first_iterable_element(s: set[T] | Iterable[T]) -> T:
    for e in s:
        return e


def _check_neg_risk_coherent_question_ids(
    all_quintets: Iterable[MarketIdQuintet],
) -> None:
    all_qids = [int(q[2][-2:], 16) for q in all_quintets]

    if not all_qids:
        raise PolyPyException("`all_market_quintets` is empty.")

    # check if all_qids consists of only consecutive numbers
    if not max(all_qids) - min(all_qids) + 1 == len(all_qids) == len(set(all_qids)):
        # todo this check might be too restrictive
        raise PolyPyException(
            f"`all_market_quintets` question_ids are not consecutive: {all_qids} for {all_quintets}."
        )

    # todo additional checks (e.g., all closed markets included)? (constraint: without REST calls)


def _check_equal_neg_risk_market_id(
    quintets: Iterable[MarketIdQuintet | tuple],
) -> str:
    if len(nrm_id_set := {q[1] for q in quintets}) != 1:
        raise PolyPyException(f"`negative_risk_market_id`s do not match: {quintets}.")

    if "" in nrm_id_set or None in nrm_id_set:
        raise PolyPyException(f"Non-negative risk market in quintets={quintets}")

    return _first_iterable_element(quintets)[1]


def _check_complete_subset_condition_ids(
    new_condition_ids: set[str] | KeysView[str], seen_condition_ids: set[str]
):
    if missing_ids := (seen_condition_ids - new_condition_ids):
        raise PolyPyException(f"Missing condition_ids {missing_ids}.")


class AugmentedConversionCache(ConversionCacheProtocol):
    """This implementation can be used in multithreading but not in multiprocessing contexts.

    Each Cache should only be bound to one and exactly one PositionManager.
    Exception to this: maintain one separate PositionManager which collects all in retrospect
    additionally added outcomes/conditions because, e.g., the concrete strategy does not care about those.
    """

    def __init__(self, endpoint_gamma: str | ENDPOINT, max_size: int | None = None):
        self.endpoint_gamma = endpoint_gamma
        self.max_size = math.inf if max_size is None else max_size

        self.caches: dict[str, _NegRiskConversionCache] = {}
        self.lock = RLock()

    def __contains__(self, neg_risk_market_id: str) -> bool:
        return neg_risk_market_id in self.caches

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

        _check_complete_subset_condition_ids(lookup.keys(), cache.seen_condition_ids)

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
        _check_neg_risk_coherent_question_ids(all_market_quintets)

        nrm = _check_equal_neg_risk_market_id(all_market_quintets)
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

    def _update_by_condition_id(
        self, size: NumericAlias | None, condition_id: str
    ) -> tuple[tuple[str, bool], tuple[Decimal, list[str]]]:
        event_closed, all_quintets = get_neg_risk_markets(
            endpoint_gama=self.endpoint_gamma,
            include_closed=True,
            condition_ids=condition_id,
            token_ids=None,
            market_slugs=None,
        )

        with self.lock:
            re_size, token_1_ids = self.update(size, all_quintets)
            nrm = all_quintets[0][1]

            # delete closed event
            if event_closed:
                del self.caches[nrm]

            return (nrm, event_closed), (re_size, token_1_ids)

    def pull(self) -> tuple[list[tuple[str, bool]], list[tuple[Decimal, list[str]]]]:
        ret = []
        closed_neg_market_ids = []

        with self.lock:
            condition_ids = {
                _first_iterable_element(cache.seen_condition_ids)
                for cache in self.caches.values()
            }

            if not len(condition_ids):
                return [], []

            events_closed, quintet_groups = get_neg_risk_markets(
                endpoint_gama=self.endpoint_gamma,
                include_closed=True,
                condition_ids=condition_ids,
            )

            for event_c, quintets in zip(events_closed, quintet_groups):
                _check_equal_neg_risk_market_id(quintets)
                size, token_1_ids = self.update(
                    size=Decimal(0), all_market_quintets=quintets
                )
                ret.append((size, token_1_ids))

                closed_neg_market_ids.append((quintets[0][1], event_c))

            for closed_nrm, event_c in closed_neg_market_ids:
                if event_c:
                    # if market is closed, no active conversions possible anymore,
                    # though claiming positions still possible (this is managed separately in PositionManager)
                    del self.caches[closed_nrm]

            return closed_neg_market_ids, ret

    def _check_condition_neg_risk_id(
        self, condition_id: str | None, neg_risk_market_id: str | None
    ) -> str:
        if is_all_none(condition_id, neg_risk_market_id):
            raise PolyPyException(
                "At least one of `condition_id` or `neg_rist_market_id` must not be None."
            )

        if condition_id is not None and all(
            condition_id not in cache.seen_condition_ids
            for cache in self.caches.values()
        ):
            raise PolyPyException(f"{condition_id} not in {self.__class__.__name__}.")

        if neg_risk_market_id is not None and neg_risk_market_id not in self:
            raise PolyPyException(
                f"{neg_risk_market_id} not in {self.__class__.__name__}."
            )

        # noinspection PyTypeChecker
        if (
            condition_id is not None
            and neg_risk_market_id is not None
            and condition_id not in self.caches[neg_risk_market_id].seen_condition_ids
        ):
            raise PolyPyException(f"{condition_id} not in {neg_risk_market_id} cache.")

        if condition_id is None:
            # noinspection PyTypeChecker
            condition_id = _first_iterable_element(self.caches[neg_risk_market_id])

        return condition_id

    def pull_by_id(
        self, condition_id: str | None, neg_risk_market_id: str | None
    ) -> tuple[tuple[str, bool], tuple[Decimal, list[str]]]:
        condition_id = self._check_condition_neg_risk_id(
            condition_id, neg_risk_market_id
        )
        return self._update_by_condition_id(None, condition_id)
