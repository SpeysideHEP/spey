import logging
import os
import re
import sys
import textwrap
from typing import Any, Callable, Dict, List, Literal, Optional, Text, Tuple, Union

import numpy as np
import pkg_resources
from semantic_version import SimpleSpec, Version

from spey.base import BackendBase, ConverterBase
from spey.combiner import UnCorrStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.system import logger
from spey.system.exceptions import PluginError
from spey.system.webutils import ConnectionError, check_updates, get_bibtex

from ._version import __version__
from .about import about
from .utils import ExpectationType

__all__ = [
    "version",
    "StatisticalModel",
    "UnCorrStatisticsCombiner",
    "ExpectationType",
    "AvailableBackends",
    "get_backend",
    "get_backend_metadata",
    "reset_backend_entries",
    "BackendBase",
    "ConverterBase",
    "about",
    "math",
    "check_updates",
    "get_backend_bibtex",
    "cite",
    "set_log_level",
]


def __dir__():
    return __all__


logger.init(LoggerStream=sys.stdout)
log = logging.getLogger("Spey")
log.setLevel(logging.INFO)


def set_log_level(level: Literal[0, 1, 2, 3]) -> None:
    """
    Set log level for spey

    Args:
        level (``int``): log level

            * 0: error
            * 1: warning
            * 2: info
            * 3: debug
    """
    log_dict = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    log.setLevel(log_dict[level])


def version() -> Text:
    """
    Version of ``spey`` package

    Returns:
        ``Text``: version in X.Y.Z format
    """
    return __version__


def _get_backend_entrypoints() -> Dict[Text, pkg_resources.EntryPoint]:
    """Collect plugin entries"""
    return {
        entry.name: entry
        for entry in pkg_resources.iter_entry_points("spey.backend.plugins")
    }


_backend_entries: Dict[Text, pkg_resources.EntryPoint] = _get_backend_entrypoints()
# ! Preinitialise backends, it might be costly to scan the system everytime


def reset_backend_entries() -> None:
    """Scan the system for backends and reset the entries"""
    _backend_entries = _get_backend_entrypoints()


def AvailableBackends() -> List[Text]:  # pylint: disable=C0103
    """
    Returns a list of available backends. The default backends are automatically installed
    with ``spey`` package. To enable other backends, please see the relevant section
    in the documentation.

    .. note::

        This function does not validate the backend. For backend validation please use
        :func:`~spey.get_backend` function.

    Returns:
        ``List[Text]``: list of names of available backends.
    """
    return [*_backend_entries.keys()]


def get_backend(name: Text) -> Callable[[Any], StatisticalModel]:
    """
    Statistical model backend retreiver. Available backend names can be found via
    :func:`~spey.AvailableBackends` function.

    Args:
        name (``Text``): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Raises:
        :obj:`~spey.system.exceptions.PluginError`: If the backend is not available
          or the available backend requires different version of ``spey``.
        :obj:`AssertionError`: If the backend does not have necessary metadata.

    Returns:
        ``Callable[[Any, ...], StatisticalModel]``:
        A callable function that takes backend specific arguments and two additional
        keyword arguments ``analysis`` (which is a unique identifier of analysis name as :obj:`str`)
        and ``xsection`` (which is cross section value with a.u.). Details about the function can be
        found in :func:`~spey.statistical_model_wrapper`. This wrapper
        returns a :obj:`~spey.StatisticalModel` object.

    Example:

    .. code-block:: python3
        :linenos:

        >>> import spey; import numpy as np
        >>> stat_wrapper = spey.get_backend("default_pdf.uncorrelated_background")

        >>> data = np.array([1])
        >>> signal = np.array([0.5])
        >>> background = np.array([2.])
        >>> background_unc = np.array([1.1])

        >>> stat_model = stat_wrapper(
        ...     signal_yields=signal,
        ...     background_yields=background,
        ...     data=data,
        ...     covariance_matrix=background_unc,
        ...     analysis="simple_sl",
        ...     xsection=0.123
        ... )
        >>> stat_model.exclusion_confidence_level()

    .. note::

        The documentation of the ``stat_wrapper`` defined above includes the docstring
        of the backend as well. Hence typing ``stat_wrapper?`` in terminal will result with
        complete documentation for the :func:`~spey.statistical_model_wrapper` and
        the backend it self which is in this particular example
        :obj:`~spey.backends.simplifiedlikelihood_backend.interface.SimplifiedLikelihoodInterface`.
    """
    backend = _backend_entries.get(name, False)

    if backend:
        statistical_model = backend.load()

        assert hasattr(
            statistical_model, "spey_requires"
        ), "Backend does not have `'spey_requires'` attribute."

        if getattr(statistical_model, "name", False):
            assert (
                statistical_model.name == name
            ), "The identity of the statistical model is wrongly set."
        else:
            setattr(statistical_model, "name", name)

        if Version(version()) not in SimpleSpec(statistical_model.spey_requires):
            raise PluginError(
                f"The backend {name}, requires spey version {statistical_model.spey_requires}. "
                f"However the current spey version is {__version__}."
            )

        # Initialise converter base models
        if ConverterBase in statistical_model.mro():
            statistical_model = statistical_model()

        return statistical_model_wrapper(statistical_model)

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )


def get_backend_metadata(name: Text) -> Dict[Text, Any]:
    """
    Retreive metadata about the backend. This includes citation information,
    doi, author names etc. Available backend names can be found via
    :func:`~spey.AvailableBackends` function.

    Args:
        name (``Text``): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Raises:
        ~spey.system.exceptions.PluginError: If the backend does not exists.

    Returns:
        ``Dict[Text, Text]``:
        Metadata about the backend.

    Example:

    .. code-block:: python3

        >>> spey.get_backend_metadata("default_pdf.third_moment_expansion")

    will return the following

    .. code-block:: python3

        >>> {'name': 'default_pdf.third_moment_expansion',
        ... 'author': 'SpeysideHEP',
        ... 'version': '0.0.1',
        ... 'spey_requires': '0.0.1',
        ... 'doi': ['10.1007/JHEP04(2019)064'],
        ... 'arXiv': ['1809.05548']}
    """
    backend = _backend_entries.get(name, False)

    if backend:
        statistical_model = backend.load()
        return {
            "name": getattr(statistical_model, "name", "__unknown_model__"),
            "author": getattr(statistical_model, "author", "__unknown_author__"),
            "version": getattr(statistical_model, "version", "__unknown_version__"),
            "spey_requires": statistical_model.spey_requires,
            "doi": list(getattr(statistical_model, "doi", [])),
            "arXiv": list(getattr(statistical_model, "arXiv", [])),
            "zenodo": list(getattr(statistical_model, "zenodo", [])),
        }

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )


def get_backend_bibtex(name: Text) -> Dict[Text, List[Text]]:
    """
    Retreive BibTex entry for backend plug-in if available.

    The bibtext entries are retreived both from Inspire HEP, doi.org and zenodo.
    If the arXiv number matches the DOI the output will include two versions
    of the same reference. If backend does not include an arXiv or DOI number
    it will return an empty list.

    .. versionadded:: 0.1.6

    Args:
        name (``Text``): backend identifier. This backend refers to different packages
          that prescribes likelihood function.

    Returns:
        ``Dict[Text, List[Text]]``:
        BibTex entries for the backend. Keywords include inspire, doi.org and zenodo.

        .. versionchanged:: 0.1.7

            In the previous version, function was returning ``List[Text]`` now it returns
            a ``Dict[Text, List[Text]]`` indicating the source of BibTeX entry.

    """
    # pylint: disable=import-outside-toplevel, W1203
    out = {"inspire": [], "doi.org": [], "zenodo": []}
    meta = get_backend_metadata(name)

    try:
        for arxiv_id in meta.get("arXiv", []):
            tmp = get_bibtex("inspire/arxiv", arxiv_id)
            if tmp != "":
                out["inspire"].append(textwrap.indent(tmp, " " * 4))
            else:
                log.debug(f"Can not find {arxiv_id} in Inspire")
        for doi in meta.get("doi", []):
            tmp = get_bibtex("inspire/doi", doi)
            if tmp == "":
                log.debug(f"Can not find {doi} in Inspire, looking at doi.org")
                tmp = get_bibtex("doi", doi)
                if tmp != "":
                    out["doi.org"].append(tmp)
                else:
                    log.debug(f"Can not find {doi} in doi.org")
            else:
                out["inspire"].append(textwrap.indent(tmp, " " * 4))
        for zenodo_id in meta.get("zenodo", []):
            tmp = get_bibtex("zenodo", zenodo_id)
            if tmp != "":
                out["zenodo"].append(textwrap.indent(tmp, " " * 4))
            else:
                log.debug(f"{zenodo_id} is not a valid zenodo identifier")
    except ConnectionError as err:
        log.error("Can not connect to the internet. Please check your connection.")
        log.debug(str(err))
        return out

    return out


def cite() -> List[Text]:
    """Retreive BibTex information for Spey"""
    try:
        arxiv = textwrap.indent(get_bibtex("inspire/arxiv", "2307.06996"), " " * 4)
        zenodo = get_bibtex("zenodo", "10156353")
        linker = re.search("@software{(.+?),\n", zenodo).group(1)
        zenodo = textwrap.indent(zenodo.replace(linker, "spey_zenodo"), " " * 4)
        return arxiv + "\n\n" + zenodo
    except ConnectionError as err:
        log.error("Can not connect to the internet. Please check your connection.")
        log.debug(str(err))
        return ""


if os.environ.get("SPEY_CHECKUPDATE", "ON").upper() != "OFF":
    check_updates()
