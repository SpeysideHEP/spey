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

* Hashable scalars (``int``, ``float``, ``str``, ...) -> used directly.
* ``numpy.ndarray`` -> ``(shape, dtype, sha256_digest)`` tuple built from a
  SHA-256 hash of the raw buffer so the key captures the full content
  without retaining a copy of the array bytes.
* ``list`` / ``tuple`` -> recursively tokenised element-by-element.
* ``dict`` -> sorted-items tuple of ``(key_token, value_token)`` pairs.
* Anything else -> falls back to ``repr()``.

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

Per-instance caching
--------------------
When ``cache_results`` is applied to an instance method (first parameter
named ``self``), it automatically returns a
:class:`_PerInstanceCacheDescriptor`.  This can also be requested
explicitly via ``per_instance=True``.  The descriptor installs a dedicated
cached callable on each instance the first time the method is accessed,
ensuring that two distinct instances of the same class (or of different
subclasses) can never share a cache entry.  ``self`` is never included in
the cache key and no reference to the instance is held in the cache
dictionary.

For classes that use ``__slots__`` without ``__dict__``, the descriptor
falls back to a descriptor-level mapping keyed by ``id(obj)``.  If the
instance supports weak references, cleanup is automatic; otherwise the
mapping entry persists for the lifetime of the descriptor (bounded and
equivalent to the prior behaviour).

.. currentmodule:: spey.system.cache

.. autosummary::
    :toctree: ../_generated/

    cache_results
    _PerInstanceCacheDescriptor
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import logging
import threading
import weakref
from collections import deque
from functools import wraps
from typing import Any, Callable

import numpy as np

__all__ = ["cache_results"]

log = logging.getLogger("Spey")

# Internal sentinel -- its identity (not value) marks an in-flight computation.
_COMPUTING = object()


def _make_token(obj: Any) -> Any:
    """Return a hashable, content-based token for *obj*.

    The token is used as (part of) the dictionary key for the result cache.
    It must satisfy two properties:

    1. **Deterministic** -- identical contents produce identical tokens.
    2. **Hashable** -- the token can be used inside a ``dict`` key tuple.
    """
    if isinstance(obj, np.ndarray):
        # Hash the raw buffer content (O(n) time, O(1) space) instead of
        # storing a full copy via tobytes() (O(n) time, O(n) space).
        buf = np.ascontiguousarray(obj)
        digest = hashlib.sha256(buf.data).digest()
        return (obj.shape, obj.dtype.str, digest)
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


def _has_self_param(fn: Callable) -> bool:
    """Return True if *fn* looks like an unbound instance method (first param is 'self')."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    params = list(sig.parameters.keys())
    return bool(params) and params[0] == "self"


class _PerInstanceCacheDescriptor:
    """
    Non-data descriptor that provides per-instance caching for instance methods.

    On the first ``instance.method`` attribute lookup the descriptor builds a
    dedicated cached callable whose internal cache storage is private to that
    instance.  The wrapper is stored in ``instance.__dict__[method_name]``
    so that Python's attribute-lookup rules (instance dict beats non-data
    descriptor) bypass the descriptor on all subsequent accesses -- zero
    overhead after the first call.

    For classes that use ``__slots__`` without ``__dict__``, the descriptor
    falls back to a descriptor-level mapping keyed by ``id(obj)``.  If the
    instance supports weak references a callback is registered to clean up
    the mapping entry when the instance is garbage-collected; otherwise the
    entry persists (bounded, and equivalent to the prior behaviour where
    ``self`` was serialised into a closure-level cache key).

    Two distinct instances therefore maintain completely independent caches
    even when they invoke the method with identical arguments, and instances
    of different subclasses that happen to share the same method name are
    equally isolated.

    ``self`` is never serialised into cache keys.  The cache dict holds only
    non-self argument tokens as keys and computed results as values; no
    reference to the instance is stored in the cache dictionary.

    This class is an implementation detail of :func:`cache_results`; obtain
    one via ``cache_results(..., per_instance=True)`` or by decorating a
    method whose first parameter is ``self`` (auto-detected).
    """

    def __init__(self, fn: Callable, maxsize: int, copy_on_return: bool) -> None:
        self._fn = fn
        self._maxsize = maxsize
        self._copy_on_return = copy_on_return
        # Signature without 'self' for canonical key building (includes defaults).
        try:
            full_sig = inspect.signature(fn)
            params = list(full_sig.parameters.values())[1:]  # skip self
            self._sig = inspect.Signature(params)
        except (ValueError, TypeError):
            self._sig = None
        # Method name used as the instance-dict key; updated by __set_name__.
        self._name: str = fn.__name__
        # Propagate function metadata so the descriptor looks like the original.
        self.__wrapped__ = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__annotations__ = getattr(fn, "__annotations__", {})
        # Fallback storage for __slots__-only classes (no __dict__).
        self._fallback_lock = threading.Lock()
        self._fallback_wrappers: dict[int, Callable] = {}
        self._fallback_refs: dict[int, weakref.ref] = {}

    def __set_name__(self, _owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, _objtype: type = None) -> Callable:
        if obj is None:
            # When accessed via class, return the ORIGINAL FUNCTION without caching.
            return self._fn  # type: ignore[return-value]
        # Fast path: try instance __dict__ (available for most classes).
        try:
            inst_dict = obj.__dict__
        except AttributeError:
            return self._get_fallback(obj)
        wrapper = inst_dict.get(self._name)
        if wrapper is None:
            wrapper = self._build_wrapper(obj)
            # Storing under the method name makes subsequent lookups land in
            # the instance dict directly (non-data descriptor rule), so the
            # descriptor overhead is paid exactly once per instance.
            inst_dict[self._name] = wrapper
        return wrapper

    def _get_fallback(self, obj: Any) -> Callable:
        """Fallback for ``__slots__``-only classes without ``__dict__``."""
        obj_id = id(obj)
        with self._fallback_lock:
            wrapper = self._fallback_wrappers.get(obj_id)
            if wrapper is not None:
                return wrapper
            wrapper = self._build_wrapper(obj)
            self._fallback_wrappers[obj_id] = wrapper
            # Register weak-ref cleanup if the instance supports it.
            try:
                ref = weakref.ref(obj, self._make_fallback_cleanup(obj_id))
                self._fallback_refs[obj_id] = ref
            except TypeError:
                pass  # Not weakly referenceable; entry persists.
            return wrapper

    def _make_fallback_cleanup(self, obj_id: int) -> Callable:
        """Return a weak-ref callback that removes the fallback entry for *obj_id*."""

        def _cleanup(_ref: weakref.ref) -> None:  # noqa: ARG001
            with self._fallback_lock:
                self._fallback_wrappers.pop(obj_id, None)
                self._fallback_refs.pop(obj_id, None)

        return _cleanup

    def _build_wrapper(self, obj: Any) -> Callable:
        """Create a standalone cached callable bound to *obj*."""
        fn = self._fn
        maxsize = self._maxsize
        copy_on_return = self._copy_on_return
        _sig = self._sig

        cache: dict = {}
        lock = threading.Lock()
        events: dict = {}
        order: deque = deque()

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _sig is not None:
                try:
                    bound = _sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    key = _build_key((), dict(bound.arguments))
                except TypeError:
                    key = _build_key(args, kwargs)
            else:
                key = _build_key(args, kwargs)

            with lock:
                if key in cache:
                    value = cache[key]
                    if value is not _COMPUTING:
                        log.debug("cache hit for %s", fn.__name__)
                        return copy.deepcopy(value) if copy_on_return else value
                    event = events[key]
                else:
                    event = threading.Event()
                    events[key] = event
                    cache[key] = _COMPUTING
                    order.append(key)
                    event = None

            if event is not None:
                event.wait()
                with lock:
                    if key in cache and cache[key] is not _COMPUTING:
                        return copy.deepcopy(cache[key]) if copy_on_return else cache[key]
                return wrapper(*args, **kwargs)

            try:
                result = fn(obj, *args, **kwargs)
            except BaseException:
                with lock:
                    if cache.get(key) is _COMPUTING:
                        del cache[key]
                        order.remove(key)
                    evt = events.pop(key, None)
                if evt is not None:
                    evt.set()
                raise

            with lock:
                cache[key] = result
                if maxsize is not None:
                    while len(cache) > maxsize:
                        oldest = order.popleft()
                        cache.pop(oldest, None)
                evt = events.pop(key, None)
            if evt is not None:
                evt.set()

            return copy.deepcopy(result) if copy_on_return else result

        def cache_clear() -> None:
            """Remove all cached entries for this instance."""
            with lock:
                cache.clear()
                order.clear()
                events.clear()

        def cache_info() -> dict:
            """Return a snapshot of the cache state for this instance."""
            with lock:
                return {"size": len(cache), "maxsize": maxsize}

        def get_cache() -> dict:
            """Return a copy of the current cache contents (for testing/debugging)."""
            with lock:
                return copy.deepcopy(cache)

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        wrapper.get_cache = get_cache
        return wrapper


def cache_results(
    func: Callable = None,
    *,
    maxsize: int = 128,
    copy_on_return: bool = True,
    per_instance: bool = False,
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
        per_instance (``bool``, default ``False``):
          When ``True``, return a :class:`_PerInstanceCacheDescriptor` instead
          of a plain wrapper.  The descriptor installs a dedicated cache on
          each instance the first time the method is accessed, so two distinct
          instances (even of the same class) never share cache entries.  Use
          this when decorating instance methods that must remain isolated
          across objects.  Must be ``False`` when decorating plain functions.

          .. note::

              Instance methods whose first parameter is ``self`` are
              **auto-detected** and routed to the per-instance descriptor
              even when this flag is ``False``.

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

        Per-instance caching for a method (typically applied via
        ``__init_subclass__`` rather than directly):

        .. code-block:: python

            @cache_results(per_instance=True)
            def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
                ...

    """

    def decorator(fn: Callable) -> Callable:
        # Explicit per_instance=True or auto-detect: first parameter named 'self'.
        if per_instance or _has_self_param(fn):
            return _PerInstanceCacheDescriptor(fn, maxsize, copy_on_return)

        cache: dict = {}
        lock = threading.Lock()
        # Per-key events: signals waiters that the computation is done.
        events: dict = {}
        # FIFO eviction order (deque for O(1) popleft).
        order: deque = deque()

        try:
            _sig = inspect.signature(fn)
        except (ValueError, TypeError):
            _sig = None

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _sig is not None:
                try:
                    bound = _sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    key = _build_key((), dict(bound.arguments))
                except TypeError:
                    key = _build_key(args, kwargs)
            else:
                key = _build_key(args, kwargs)

            with lock:
                if key in cache:
                    value = cache[key]
                    if value is not _COMPUTING:
                        # Cache hit -- return immediately.
                        log.debug("cache hit for %s", fn.__name__)
                        return copy.deepcopy(value) if copy_on_return else value
                    # Another thread is computing this key -- grab its event.
                    event = events[key]
                else:
                    # First caller for this key -- mark it as in-flight.
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
                # The computing thread failed (sentinel removed) -- fall through
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
                        oldest = order.popleft()
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
                events.clear()

        def cache_info() -> dict:
            """Return a snapshot of the cache state."""
            with lock:
                return {"size": len(cache), "maxsize": maxsize}

        def get_cache() -> dict:
            """Return a copy of the current cache contents (for testing/debugging)."""
            with lock:
                return copy.deepcopy(cache)

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        wrapper.get_cache = get_cache
        return wrapper

    # Support both @cache_results and @cache_results(...) forms.
    if func is not None:
        return decorator(func)
    return decorator
