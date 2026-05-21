import logging
from dataclasses import dataclass
from typing import Any, List, Dict

from .core import fit

__all__ = ["fit"]

log = logging.getLogger("Spey")


@dataclass
class ValidateOpts:
    """
    Whitelist filter for optimiser keyword options.

    Used by :mod:`spey.optimizer.scipy_tools` and
    :mod:`spey.optimizer.minuit_tools` to enforce that callers only pass options
    recognised by the chosen minimiser. Unknown keys are dropped (with a
    warning), required keys are checked, and keys explicitly listed under
    ``remove_list`` are stripped after validation (typically because they are
    spey-internal hints, e.g. ``poi_index``).

    Attributes:
        opt_list (``List[str]``): Keys that are accepted and forwarded.
        must_list (``List[str]``, default ``[]``): Keys that must be present;
          missing keys raise :class:`AssertionError`.
        remove_list (``List[str]``, default ``[]``): Keys that are accepted but
          stripped from the returned dict (silently consumed).
    """

    opt_list: List[str]
    must_list: List[str] = None
    remove_list: List[str] = None

    def __post_init__(self):
        self.must_list = [] if self.must_list is None else self.must_list
        self.remove_list = [] if self.remove_list is None else self.remove_list

    def __call__(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter ``options`` in place and return it.

        Args:
            options (``Dict[str, Any]``): Mapping of option names to values.

        Raises:
            ``AssertionError``: If any key in :attr:`must_list` is absent.

        Returns:
            ``Dict[str, Any]``: The same dict with unknown / ``remove_list``
            keys removed.
        """
        if not all(k in options for k in self.must_list):
            raise AssertionError("Options should include " + ", ".join(self.must_list))
        for key in list(options):
            if key not in self.opt_list + self.must_list + self.remove_list:
                log.warning("%s is not a valid option.", key)
                options.pop(key)
            if key in self.remove_list:
                options.pop(key)
        return options
