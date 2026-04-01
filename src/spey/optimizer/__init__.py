import logging
from dataclasses import dataclass
from typing import Any, List, Dict

from .core import fit

__all__ = ["fit"]

log = logging.getLogger("Spey")


@dataclass
class ValidateOpts:
    """Validate optimiser options"""

    opt_list: List[str]
    must_list: List[str] = None
    remove_list: List[str] = None

    def __post_init__(self):
        self.must_list = [] if self.must_list is None else self.must_list
        self.remove_list = [] if self.remove_list is None else self.remove_list

    def __call__(self, options: Dict[str, Any]) -> Dict[str, Any]:
        if not all(k in options for k in self.must_list):
            raise AssertionError("Options should include " + ", ".join(self.must_list))
        for key in list(options):
            if key not in self.opt_list + self.must_list + self.remove_list:
                log.warning("%s is not a valid option.", key)
                options.pop(key)
            if key in self.remove_list:
                options.pop(key)
        return options
