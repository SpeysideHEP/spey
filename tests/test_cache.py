"""Tests for spey.system.cache — correctness, isolation, eviction, thread safety."""

import gc
import threading
import weakref

import numpy as np
import pytest

from spey.system.cache import cache_results, _PerInstanceCacheDescriptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

call_count = 0


def _reset_counter():
    global call_count
    call_count = 0


class DummyBackend:
    """Minimal class to test per-instance caching."""

    def __init__(self, value):
        self.value = value

    @cache_results(maxsize=8, copy_on_return=True, per_instance=True)
    def compute(self, x, y=1):
        global call_count
        call_count += 1
        return self.value * x + y


class AutoDetectedBackend:
    """Per-instance caching via auto-detection (no explicit per_instance=True)."""

    def __init__(self, value):
        self.value = value

    @cache_results(maxsize=8, copy_on_return=True)
    def compute(self, x, y=1):
        global call_count
        call_count += 1
        return self.value * x + y


# ---------------------------------------------------------------------------
# 1. Cache hits and misses
# ---------------------------------------------------------------------------


class TestHitsAndMisses:
    def test_basic_hit(self):
        _reset_counter()
        a = DummyBackend(10)
        assert a.compute(3) == 31  # 10*3 + 1
        assert a.compute(3) == 31  # cache hit
        assert call_count == 1

    def test_different_args_miss(self):
        _reset_counter()
        a = DummyBackend(10)
        a.compute(1)
        a.compute(2)
        assert call_count == 2

    def test_kwarg_normalisation(self):
        """Calling f(2) and f(x=2) should share the same cache entry."""
        _reset_counter()
        a = DummyBackend(10)
        a.compute(2)
        a.compute(x=2)
        assert call_count == 1

    def test_default_kwarg_normalisation(self):
        """f(2) and f(2, y=1) should share the same cache entry."""
        _reset_counter()
        a = DummyBackend(10)
        a.compute(2)
        a.compute(2, y=1)
        assert call_count == 1

    def test_plain_function_cache(self):
        counter = {"n": 0}

        @cache_results(maxsize=4, copy_on_return=False)
        def add(a, b):
            counter["n"] += 1
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3
        assert counter["n"] == 1
        assert add(2, 3) == 5
        assert counter["n"] == 2

    def test_numpy_array_cache(self):
        counter = {"n": 0}

        @cache_results(maxsize=4, copy_on_return=True)
        def process(arr):
            counter["n"] += 1
            return arr.sum()

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])  # same content, different object
        c = np.array([4.0, 5.0, 6.0])

        assert process(a) == 6.0
        assert process(b) == 6.0  # cache hit (same content)
        assert counter["n"] == 1
        assert process(c) == 15.0  # miss
        assert counter["n"] == 2


# ---------------------------------------------------------------------------
# 2. Instance isolation
# ---------------------------------------------------------------------------


class TestInstanceIsolation:
    def test_per_instance_explicit(self):
        """Two DummyBackend instances must have independent caches."""
        _reset_counter()
        a = DummyBackend(10)
        b = DummyBackend(20)

        assert a.compute(3) == 31  # 10*3 + 1
        assert b.compute(3) == 61  # 20*3 + 1
        assert call_count == 2

        # Mutating b's cache should not affect a
        b.compute.cache_clear()
        assert a.compute(3) == 31  # still cached
        assert call_count == 2  # no new call

    def test_per_instance_auto_detected(self):
        """Auto-detected per-instance: same isolation guarantees."""
        _reset_counter()
        a = AutoDetectedBackend(10)
        b = AutoDetectedBackend(20)

        assert a.compute(3) == 31
        assert b.compute(3) == 61
        assert call_count == 2

        b.compute.cache_clear()
        assert a.compute(3) == 31
        assert call_count == 2

    def test_no_shared_mutable_state(self):
        """Mutating a returned value must not corrupt the cache."""
        a = DummyBackend(1)

        @cache_results(maxsize=4, copy_on_return=True)
        def make_list(n):
            return [n]

        result1 = make_list(5)
        result1.append(99)  # mutate the returned copy
        result2 = make_list(5)
        assert result2 == [5]  # cache still returns the original

    def test_self_not_in_cache_key(self):
        """Verify that `self` is never serialised into the cache key."""
        a = DummyBackend(10)
        a.compute(1)
        raw_cache = a.compute.get_cache()
        for key in raw_cache:
            # key is (args_token, kwargs_token)
            flat = str(key)
            assert "DummyBackend" not in flat
            assert "object at" not in flat

    def test_auto_detect_creates_descriptor(self):
        """A method with 'self' should produce a _PerInstanceCacheDescriptor."""
        assert isinstance(
            AutoDetectedBackend.__dict__["compute"], _PerInstanceCacheDescriptor
        )


# ---------------------------------------------------------------------------
# 3. Cache eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_maxsize_eviction(self):
        _reset_counter()
        a = DummyBackend(1)  # maxsize=8

        # Fill the cache beyond maxsize
        for i in range(10):
            a.compute(i)
        assert call_count == 10

        info = a.compute.cache_info()
        assert info["size"] <= 8

        # Oldest entries (0, 1) should have been evicted
        _reset_counter()
        a.compute(0)
        assert call_count == 1  # had to recompute

    def test_cache_clear(self):
        _reset_counter()
        a = DummyBackend(1)
        a.compute(1)
        a.compute(2)
        assert call_count == 2

        a.compute.cache_clear()
        assert a.compute.cache_info()["size"] == 0

        a.compute(1)
        assert call_count == 3  # had to recompute

    def test_plain_function_eviction(self):
        counter = {"n": 0}

        @cache_results(maxsize=3, copy_on_return=False)
        def square(x):
            counter["n"] += 1
            return x * x

        for x in range(5):
            square(x)
        assert counter["n"] == 5
        assert square.cache_info()["size"] == 3

        # Oldest (0, 1) evicted; 2 might be evicted too depending on order
        counter["n"] = 0
        square(4)  # should be cached (most recent)
        assert counter["n"] == 0


# ---------------------------------------------------------------------------
# 4. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_same_key(self):
        """Multiple threads requesting the same key should compute it exactly once."""
        compute_count = {"n": 0}
        start = threading.Event()

        @cache_results(maxsize=16, copy_on_return=False)
        def slow_add(a, b):
            import time

            time.sleep(0.05)  # simulate work so other threads queue up
            compute_count["n"] += 1
            return a + b

        results = [None] * 4
        errors = []

        def worker(idx):
            try:
                start.wait(timeout=5)
                results[idx] = slow_add(1, 2)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        start.set()  # release all threads at once
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors in threads: {errors}"
        assert all(r == 3 for r in results)
        assert compute_count["n"] == 1

    def test_concurrent_different_keys(self):
        """Different keys should be computed independently without deadlock."""
        counter = {"n": 0}
        lock = threading.Lock()

        @cache_results(maxsize=16, copy_on_return=False)
        def square(x):
            with lock:
                counter["n"] += 1
            return x * x

        results = {}
        errors = []

        def worker(x):
            try:
                results[x] = square(x)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert counter["n"] == 8
        for i in range(8):
            assert results[i] == i * i

    def test_concurrent_per_instance(self):
        """Per-instance caching under concurrent access."""
        compute_count = {"n": 0}

        class Worker:
            def __init__(self, mult):
                self.mult = mult

            @cache_results(maxsize=16, copy_on_return=False, per_instance=True)
            def work(self, x):
                compute_count["n"] += 1
                return self.mult * x

        obj = Worker(5)
        results = [None] * 4
        errors = []

        def call(idx):
            try:
                results[idx] = obj.work(10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert all(r == 50 for r in results)
        assert compute_count["n"] == 1


# ---------------------------------------------------------------------------
# 5. Memory: no self reference in cache, GC-friendly
# ---------------------------------------------------------------------------


class TestMemory:
    def test_instance_gc_after_cache_clear(self):
        """Instance should be garbage-collectible after all references are dropped."""
        obj = DummyBackend(42)
        ref = weakref.ref(obj)
        obj.compute(1)  # populate cache
        obj.compute.cache_clear()
        del obj
        gc.collect()
        # The weak reference should be dead (instance collected).
        assert ref() is None

    def test_cache_does_not_hold_self_reference(self):
        """The raw cache dict should not contain any reference to the instance."""
        obj = DummyBackend(7)
        obj.compute(1)
        raw = obj.compute.get_cache()
        # Serialise the entire cache to check for instance references
        for key, value in raw.items():
            assert not isinstance(value, DummyBackend)
            key_str = repr(key)
            assert "DummyBackend" not in key_str

    def test_copy_on_return_prevents_mutation(self):
        """With copy_on_return=True, mutating the return value must not affect the cache."""

        class Holder:
            @cache_results(maxsize=4, copy_on_return=True, per_instance=True)
            def get_list(self, n):
                return list(range(n))

        h = Holder()
        r1 = h.get_list(5)
        r1.append(999)
        r2 = h.get_list(5)
        assert r2 == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_exception_does_not_poison_cache(self):
        attempt = {"n": 0}

        @cache_results(maxsize=4, copy_on_return=False)
        def flaky(x):
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ValueError("transient failure")
            return x * 2

        with pytest.raises(ValueError):
            flaky(5)

        # Second call should succeed (sentinel was cleaned up)
        assert flaky(5) == 10
        assert attempt["n"] == 2

    def test_cache_info_and_clear(self):
        @cache_results(maxsize=10, copy_on_return=False)
        def identity(x):
            return x

        identity(1)
        identity(2)
        assert identity.cache_info() == {"size": 2, "maxsize": 10}

        identity.cache_clear()
        assert identity.cache_info() == {"size": 0, "maxsize": 10}

    def test_bare_decorator(self):
        """@cache_results without parentheses should work."""
        counter = {"n": 0}

        @cache_results
        def add(a, b):
            counter["n"] += 1
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3
        assert counter["n"] == 1

    def test_dict_and_list_args(self):
        counter = {"n": 0}

        @cache_results(maxsize=4, copy_on_return=False)
        def process(data, labels):
            counter["n"] += 1
            return sum(data.values())

        process({"a": 1, "b": 2}, [10, 20])
        process({"a": 1, "b": 2}, [10, 20])
        assert counter["n"] == 1

        process({"a": 1, "b": 3}, [10, 20])  # different dict value
        assert counter["n"] == 2

    def test_unlimited_maxsize(self):
        counter = {"n": 0}

        @cache_results(maxsize=None, copy_on_return=False)
        def ident(x):
            counter["n"] += 1
            return x

        for i in range(200):
            ident(i)
        assert counter["n"] == 200
        assert ident.cache_info()["size"] == 200

    def test_slots_class_fallback(self):
        """Per-instance caching should work on __slots__-only classes (no __dict__)."""
        counter = {"n": 0}

        class Base:
            __slots__ = ()

        class Slotted(Base):
            __slots__ = ("value",)

            def __init__(self, v):
                self.value = v

            @cache_results(per_instance=True, maxsize=8, copy_on_return=False)
            def compute(self, x):
                counter["n"] += 1
                return self.value * x

        a = Slotted(10)
        b = Slotted(20)
        assert not hasattr(a, "__dict__")  # confirm slots-only

        assert a.compute(3) == 30
        assert b.compute(3) == 60
        assert counter["n"] == 2

        # Cache hits
        assert a.compute(3) == 30
        assert counter["n"] == 2

        # Instance isolation
        b.compute.cache_clear()
        assert a.compute(3) == 30  # still cached
        assert counter["n"] == 2
