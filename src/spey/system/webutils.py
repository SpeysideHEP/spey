import logging
from typing import Literal, Text

import requests
from pkg_resources import get_distribution
from semantic_version import Version

log = logging.getLogger("Spey")

__all__ = ["get_bibtex", "check_updates"]


def __dir__():
    return __all__


def get_bibtex(
    home: Literal["inspire/arxiv", "inspire/doi", "doi", "zenodo"],
    identifier: Text,
    timeout: int = 5,
) -> Text:
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

    if requests is None:
        log.error("Unable to retreive information. Please install `requests`.")
        return ""

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
    except Exception as err:  # pylint: disable=W0718
        log.debug(str(err))
        return ""


def check_updates() -> None:
    """
    Check Spey Updates.

    Spey always checks updates when initialised. To disable this set the following

    Option if using terminal:

    .. code-block:: bash

        export SPEY_CHECKUPDATE=OFF

    Option if using python interface:

    .. code-block:: python

        import os
        os.environ["SPEY_CHECKUPDATE"]="OFF"

    """
    # pylint: disable=import-outside-toplevel, W1203
    try:
        response = requests.get("https://pypi.org/pypi/spey/json", timeout=1)
        response.encoding = "utf-8"
        pypi_info = response.json()
        pypi_version = pypi_info.get("info", {}).get("version", False)
        version = get_distribution("spey").version
        if pypi_version:
            if Version(version) < Version(pypi_version):
                log.warning(
                    f"An update is available. Current version of spey is {version}, "
                    f"available version is {pypi_version}."
                )
            elif Version(version) > Version(pypi_version):
                log.warning(
                    f"An unstable version of spey ({version}) is being used."
                    f" Latest stable version is {pypi_version}."
                )

    except Exception as err:  # pylint: disable=W0718
        # Can not retreive updates
        log.debug(str(err))
