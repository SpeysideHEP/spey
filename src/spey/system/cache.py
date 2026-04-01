"""
Thread-safe, mutable-argument-aware function result cache.

Provides :func:`cache_results`, a decorator that caches function return
values keyed on both positional and keyword arguments.  Unlike
:func:`functools.lru_cache`, it handles non-hashable inputs (NumPy arrays,
lists, dicts, etc.) by building a deterministic cache key from a deep,
content-based representation of every argument.

Cache key construction
----------------------
Each argument is converted to a *hashable token* via :func:`_make_token`:

* Hashable scalars (``int``, ``float``, ``str``, …) → used directly.
* ``numpy.ndarray`` → ``(shape, dtype, bytes)`` tuple built from the raw
  buffer (read-only view) so the key captures the full content.
* ``list`` / ``tuple`` → recursively tokenised element-by-element.
* ``dict`` → sorted-items tuple of ``(key_token, value_token)`` pairs.
* Anything else → falls back to ``repr()``.

The final key is a ``(args_token, sorted_kwargs_token)`` tuple that is
hashed by a standard Python :class:`dict`.

Thread safety
-------------
A :class:`threading.Lock` serialises reads and writes to the internal
cache dict.  To prevent *thundering-herd* duplicate computation, a
*sentinel* is inserted under the key **before** the decorated function
runs.  A per-key :class:`threading.Event` is used so that concurrent
callers block efficiently (no busy-wait) until the real result is stored.
Each unique input is computed exactly once even under contention.

.. currentmodule:: spey.system.cache

.. autosummary::
    :toctree: ../_generated/

    cache_results
"""

from __future__ import annotations

import copy
import logging
import threading
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["cache_results"]

log = logging.getLogger("Spey")

# Internal sentinel – its identity (not value) marks an in-flight computation.
_COMPUTING = object()


def _make_token(obj: Any) -> Any:
    """Return a hashable, content-based token for *obj*.

    The token is used as (part of) the dictionary key for the result cache.
    It must satisfy two properties:

    1. **Deterministic** – identical contents produce identical tokens.
    2. **Hashable** – the token can be used inside a ``dict`` key tuple.
    """
    if isinstance(obj, np.ndarray):
        # Use the raw byte content so the key reflects every element.
        return (obj.shape, obj.dtype.str, obj.tobytes())
    if isinstance(obj, dict):
        return tuple(sorted((_make_token(k), _make_token(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_make_token(item) for item in obj)
    if isinstance(obj, (set, frozenset)):
        return frozenset(_make_token(item) for item in obj)
    # Fast path: most primitives (int, float, str, None, bool, Enum) are
    # already hashable.
    try:
        hash(obj)
        return obj
    except TypeError:
        return repr(obj)


def _build_key(args: tuple, kwargs: dict) -> tuple:
    """Build the full cache key from positional and keyword arguments."""
    args_token = tuple(_make_token(a) for a in args)
    kwargs_token = tuple(
        sorted((_make_token(k), _make_token(v)) for k, v in kwargs.items())
    )
    return (args_token, kwargs_token)


def cache_results(
    func: Callable = None,
    *,
    maxsize: int = 128,
    copy_on_return: bool = True,
) -> Callable:
    """
    Decorator that caches function results for identical arguments.

    Args:
        func (``Optional[Callable]``, default ``None``): The function to wrap.
          When *cache_results* is used **without** parentheses (``@cache_results``),
          *func* is passed directly.
        maxsize (``Optional[int]``, default ``128``):
          Maximum number of entries to keep.  When the limit is reached the
          oldest entry (FIFO) is evicted.  ``None`` means unlimited.
        copy_on_return (``bool``, default ``True``):
          If ``True``, a :func:`copy.deepcopy` of the cached result is
          returned on every call.  This prevents callers from mutating the
          cached object at the cost of a copy.  Set to ``False`` when the
          result is known to be immutable or when performance is critical.

    Examples:

        Without arguments (uses defaults):

        .. code-block:: python

            @cache_results
            def expensive(x, arr):
                ...

        With arguments:

        .. code-block:: python

            @cache_results(maxsize=64, copy_on_return=False)
            def expensive(x, arr):
                ...

    """

    def decorator(fn: Callable) -> Callable:
        cache: dict = {}
        lock = threading.Lock()
        # Per-key events: signals waiters that the computation is done.
        events: dict = {}
        # Insertion-order list for FIFO eviction (only when maxsize is set).
        order: list = []

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = _build_key(args, kwargs)

            with lock:
                if key in cache:
                    value = cache[key]
                    if value is not _COMPUTING:
                        # Cache hit – return immediately.
                        log.debug("cache hit for %s", fn.__name__)
                        return copy.deepcopy(value) if copy_on_return else value
                    # Another thread is computing this key – grab its event.
                    event = events[key]
                else:
                    # First caller for this key – mark it as in-flight.
                    event = threading.Event()
                    events[key] = event
                    cache[key] = _COMPUTING
                    order.append(key)
                    event = None  # signals "I am the computing thread"

            if event is not None:
                # ---- wait for the computing thread to finish ---------------
                event.wait()
                with lock:
                    if key in cache and cache[key] is not _COMPUTING:
                        return copy.deepcopy(cache[key]) if copy_on_return else cache[key]
                # The computing thread failed (sentinel removed) – fall through
                # and recurse so this caller can become the new computing thread.
                return wrapper(*args, **kwargs)

            # ---- compute (outside lock so other keys aren't blocked) -------
            try:
                result = fn(*args, **kwargs)
            except BaseException:
                # Remove the sentinel so future calls can retry.
                with lock:
                    if cache.get(key) is _COMPUTING:
                        del cache[key]
                        order.remove(key)
                    evt = events.pop(key, None)
                if evt is not None:
                    evt.set()  # wake up waiters
                raise

            # ---- store result (under lock) ---------------------------------
            with lock:
                cache[key] = result
                # Evict oldest entries if maxsize is exceeded.
                if maxsize is not None:
                    while len(cache) > maxsize:
                        oldest = order.pop(0)
                        cache.pop(oldest, None)
                evt = events.pop(key, None)
            if evt is not None:
                evt.set()  # wake up any waiting threads

            return copy.deepcopy(result) if copy_on_return else result

        # --- public helpers on the wrapper ----------------------------------
        def cache_clear() -> None:
            """Remove all cached entries."""
            with lock:
                cache.clear()
                order.clear()

        def cache_info() -> dict:
            """Return a snapshot of the cache state."""
            with lock:
                return {"size": len(cache), "maxsize": maxsize}

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        return wrapper

    # Support both @cache_results and @cache_results(...) forms.
    if func is not None:
        return decorator(func)
    return decorator
