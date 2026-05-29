import json
import logging
import pathlib
from datetime import datetime, timezone
from platform import python_version
from typing import Literal

import requests
from semantic_version import SimpleSpec, Version

from spey._version import __version__

log = logging.getLogger("Spey")

__all__ = ["get_bibtex", "check_updates", "ConnectionError"]


def __dir__():
    return __all__


class ConnectionError(Exception):
    """No internet connection"""


def get_bibtex(
    home: Literal["inspire/arxiv", "inspire/doi", "doi", "zenodo"],
    identifier: str,
    timeout: int = 5,
) -> str:
    """
    Retreive BibTex information from InspireHEP, Zenodo or DOI.org

    Args:
        home (``Text``): Home location for the identifier.

            * ``"inspire/arxiv"``: retreive information from inspire using
                arXiv number.
            * ``"inspire/doi"``: retreive information from inspore using
                doi number.
            * ``"doi"``: retreive information from doi.org
            * ``"zenodo"``: retreive information from zenodo using zenodo
                identifier.

        identifier (``Text``): web identifier
        timeout (``int``, default ``5``): time out.

    Returns:
        ``Text``:
        Bibtex entry.
    """
    # pylint: disable=W1203

    home_loc = {
        "inspire/arxiv": "https://inspirehep.net/api/arxiv/%s",
        "inspire/doi": "https://inspirehep.net/api/doi/%s",
        "doi": "https://doi.org/%s",
        "zenodo": "https://zenodo.org/api/records/%s",
    }

    try:
        response = requests.get(
            home_loc[home] % identifier,
            headers={"accept": "application/x-bibtex"},
            timeout=timeout,
        )
        if not 200 <= response.status_code <= 299:
            log.debug(f"HTML Status Code: {response.status_code}")
            return ""
        response.encoding = "utf-8"
        return response.text
    except (
        requests.ConnectionError,
        requests.ConnectTimeout,
        requests.exceptions.ReadTimeout,
    ) as err:
        raise ConnectionError(
            "Can not retreive BibTeX information: No internet connection."
        ) from err
    except Exception as err:  # pylint: disable=W0718
        log.debug(str(err))
        return ""


_SPEY_DIR = pathlib.Path.home() / ".spey"
_CACHE_MAX_AGE_SECONDS = 86_400  # 24 hours


def _load_cached_pypi_info() -> dict:
    """
    Return cached PyPI JSON from ``~/.spey/pypi_latest_*.json`` if a file
    written within the last 24 hours exists, otherwise return ``None``.
    All filesystem errors are silently swallowed.
    """
    try:
        if not _SPEY_DIR.is_dir():
            return None
        now = datetime.now(tz=timezone.utc).timestamp()
        for cache_file in sorted(_SPEY_DIR.glob("pypi_latest_*.json"), reverse=True):
            try:
                age = now - cache_file.stat().st_mtime
                if age <= _CACHE_MAX_AGE_SECONDS:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    log.debug(
                        "Using cached PyPI info from %s (age %.0fs).",
                        cache_file.name,
                        age,
                    )
                    return data
                # File is older than 24 h — stop looking (files are sorted newest-first).
                break
            except Exception as err:  # noqa: BLE001
                log.debug("Could not read cache file %s: %s", cache_file.name, err)
    except Exception as err:  # noqa: BLE001
        log.debug("Cache directory scan failed: %s", err)
    return None


def _save_pypi_info(pypi_info: dict) -> None:
    """
    Save *pypi_info* to ``~/.spey/pypi_latest_{datetime}.json``.
    Old cache files (more than 24 h) are removed opportunistically.
    Silently skips if the directory cannot be created or written to.
    """
    try:
        _SPEY_DIR.mkdir(parents=False, exist_ok=True)
    except PermissionError as err:
        log.debug("Cannot create ~/.spey directory: %s", err)
        return
    except Exception as err:  # noqa: BLE001
        log.debug("Unexpected error creating ~/.spey: %s", err)
        return

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    cache_file = _SPEY_DIR / f"pypi_latest_{timestamp}.json"
    try:
        cache_file.write_text(json.dumps(pypi_info, indent=4), encoding="utf-8")
        log.debug("Saved PyPI info to %s.", cache_file.name)
    except Exception as err:  # noqa: BLE001
        log.debug("Could not write cache file %s: %s", cache_file.name, err)

    # Remove every other cache file so only the latest one is kept.
    try:
        for old_file in _SPEY_DIR.glob("pypi_latest_*.json"):
            if old_file != cache_file:
                try:
                    old_file.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass


def _get_pypi_info() -> dict:
    """
    Return PyPI JSON for spey, using a 24-hour on-disk cache.
    Returns ``None`` if the information cannot be obtained for any reason.
    """
    cached = _load_cached_pypi_info()
    if cached is not None:
        return cached

    try:
        response = requests.get(
            "https://pypi.org/pypi/spey/json",
            timeout=1,
            stream=False,
        )
        response.encoding = "utf-8"
        pypi_info = response.json()
    except Exception as err:  # noqa: BLE001
        log.debug("PyPI request failed: %s", err)
        return None

    _save_pypi_info(pypi_info)
    return pypi_info


def check_updates() -> None:
    """
    Check Spey Updates.

    .. versionadded:: 0.1.6

    Spey always checks updates when initialised. To disable this set the following

    Option if using terminal:

    .. code-block:: bash

        export SPEY_CHECKUPDATE=OFF

    Option if using python interface:

    .. code-block:: python

        import os
        os.environ["SPEY_CHECKUPDATE"]="OFF"

    PyPI is queried at most once per 24 hours; the response is cached in
    ``~/.spey/pypi_latest_{datetime}.json``.  If the cache directory cannot
    be created (e.g. no write permission) the function always attempts a
    live request.  Any network or filesystem error is logged at DEBUG level
    and the function returns silently without raising.
    """
    # pylint: disable=W1203
    try:
        pypi_info = _get_pypi_info()
        if pypi_info is None:
            return

        py_version = Version(python_version())
        pypi_version = pypi_info.get("info", {}).get("version", False)
        version = __version__
        if pypi_version:
            python_requires = SimpleSpec(pypi_info["info"]["requires_python"])
            log.debug(f"Curernt version {version}, latest version {pypi_version}.")
            if "beta" in Version(version).prerelease:
                log.warning(
                    f"A prerelease version of Spey is in use: {version}. "
                    f"Latest stable version is {pypi_version}."
                )
            elif Version(version) < Version(pypi_version):
                log.warning(
                    f"A newer version ({pypi_version}) of Spey is available. "
                    f"Current version is {version}."
                )
            elif Version(version) > Version(pypi_version):
                log.warning(
                    f"An unstable version of Spey ({version}) is being used."
                    f" Latest stable version is {pypi_version}."
                )
            if py_version not in python_requires:
                log.warning(
                    f"The latest version of Spey requires python{python_requires},"
                    f" however local python version is {py_version}."
                )

    except Exception as err:  # pylint: disable=W0718
        log.debug(str(err))
