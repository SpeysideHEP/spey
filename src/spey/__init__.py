import logging
import os
import re
import sys
import textwrap
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from importlib.util import find_spec
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

from semantic_version import SimpleSpec, Version

from spey.base import BackendBase, ConverterBase
from spey.combiner import UnCorrStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.system import logger
from spey.system.exceptions import AbstractModel, MissingMetaData, PluginError
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

    Log level can also be set through terminal by the following command

    .. code::

        export SPEY_LOGLEVEL=3

    value corresponds to the levels shown below.

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


def version() -> str:
    """
    Version of ``spey`` package

    Returns:
        ``Text``: version in X.Y.Z format
    """
    return __version__


def _get_entry_points(group: str, name: Optional[str] = None) -> Iterable[EntryPoint]:
    """
    Get entry points for a given group and optional name.
    Compatible with Python 3.8 → 3.13.

    Args:
        group (``Text``): entry point group name
        name (``Optional[Text]``): entry point name, if None, returns all in group
    Returns:
        ``Iterable[EntryPoint]``: list of entry points
    """
    if sys.version_info < (3, 10):
        # Python 3.8–3.9: entry_points() returns a dict-like mapping
        eps = entry_points().get(group, [])  # pylint: disable=no-member
        if name is not None:
            eps = [ep for ep in eps if ep.name == name]
    else:
        # Python 3.10+: entry_points() returns EntryPoints object with .select()
        if name is not None:
            eps = entry_points().select(group=group, name=name)
        else:
            eps = entry_points().select(group=group)
    return eps


def _get_backend_entrypoints() -> Dict[str, EntryPoint]:
    """Collect plugin entries"""
    return {entry.name: entry for entry in _get_entry_points("spey.backend.plugins")}


_backend_entries: Dict[str, EntryPoint] = _get_backend_entrypoints()
# ! Preinitialise backends, it might be costly to scan the system everytime


def reset_backend_entries() -> None:
    """Scan the system for backends and reset the entries"""
    _backend_entries.update(_get_backend_entrypoints())


def register_backend(
    model: Union[BackendBase, ConverterBase]
) -> Union[BackendBase, ConverterBase]:
    """
    A local backend registry for statistical models.

    .. versionadded:: 0.2.6

    Args:
        func (:obj:`~spey.BackendBase` or :obj:`~spey.ConverterBase`): statistical model
            object.

    Raises:
        `MissingMetaData`: If model does not include `name` and `spey_requires` metadata.
        `PluginError`: If model requires spey with different version.
        `ValueError`: If the model name is already registered.
        `AbstractModel`: If the model is abstract.

    Returns:
        :obj:`~spey.BackendBase` or :obj:`~spey.ConverterBase`: the original function wrapped
            with backend registration logic.

    **Example:**

    .. code:: python3

        >>> import spey

        >>> @spey.register_backend
        >>> class Model(spey.BackendBase):
        >>>     name = "my_local_model"
        >>>     ...

        >>> print(spey.AvailableBackends())
        >>> # ['default.correlated_background', 'default.effective_sigma',
        ... # 'default.multivariate_normal', 'default.normal', 'default.poisson',
        ... # 'default.third_moment_expansion', 'default.uncorrelated_background',
        ... # 'my_local_model']

    """
    assert issubclass(model, (BackendBase, ConverterBase)), "Invalid model structure."
    required_meta = ["spey_requires", "name"]
    if bool(getattr(model, "__abstractmethods__", False)):
        raise AbstractModel(
            "Can not register abstract models. Please fill the missing method(s): "
            + ", ".join(getattr(model, "__abstractmethods__"))
        )

    if all(hasattr(model, meta) for meta in required_meta):
        raise MissingMetaData("Required metadata missing: " + ", ".join(required_meta))

    name = getattr(model, "name", "__unknown_model__")
    if name in _backend_entries:
        raise ValueError(f"Backend name `{name}`, is already registered.")
    if name == "__unknown_model__":
        log.warning("Model does not have a name, registring as `__unknown_model__`")

    if Version(__version__) not in SimpleSpec(model.spey_requires):
        raise PluginError(
            f"The backend {name}, requires spey version {model.spey_requires}. "
            f"However the current spey version is {__version__}."
        )

    # Register the model with the backend
    _backend_entries[name] = model
    return model


def AvailableBackends() -> List[str]:  # pylint: disable=C0103
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


def get_backend(name: str) -> Callable[[Any], StatisticalModel]:
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
        >>> stat_wrapper = spey.get_backend("default.uncorrelated_background")

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
        statistical_model = (
            backend
            if isinstance(backend, (BackendBase, ConverterBase))
            else backend.load()
        )

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

        if bool(getattr(statistical_model, "__abstractmethods__", False)):
            raise AbstractModel(
                f"The backend {name} is an abstract model. Please fill the missing method(s): "
                + ", ".join(getattr(statistical_model, "__abstractmethods__"))
            )

        # Initialise converter base models
        if issubclass(statistical_model, ConverterBase):
            statistical_model = statistical_model()

        return statistical_model_wrapper(statistical_model)

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )


def get_backend_metadata(name: str) -> Dict[str, Any]:
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

        >>> spey.get_backend_metadata("default.third_moment_expansion")

    will return the following

    .. code-block:: python3

        >>> {'name': 'default.third_moment_expansion',
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


def get_backend_bibtex(name: str) -> Dict[str, List[str]]:
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


def cite() -> List[str]:
    """Retreive BibTex information for Spey"""
    try:
        arxiv = get_bibtex("inspire/arxiv", "2307.06996")
        zenodo = get_bibtex("zenodo", "10156353")
        linker = re.search("@software{(.+?),\n", zenodo)
        if linker is not None:
            zenodo = zenodo.replace(linker.group(1), "spey_zenodo")
        return arxiv + "\n\n" + zenodo
    except ConnectionError as err:
        log.error("Can not connect to the internet. Please check your connection.")
        log.debug(str(err))
        return ""


@lru_cache(10)
def log_once(msg: str, log_type: Literal["warning", "error", "info", "debug"]) -> None:
    """
    Log for every 10 messages

    Args:
        msg (``str``): message to be logged
        log_type (``str``): type of log message. ``"warning"``, ``"error"``,
          ``"info"`` or ``"debug"``.
    """
    {
        "warning": log.warning,
        "error": log.error,
        "info": log.info,
        "debug": log.debug,
    }.get(log_type, log.info)(msg)


def set_optimiser(name: str) -> None:
    """
    Set optimiser for fitting interface.

    Alternatively, optimiser can be set through terminal via

    .. code:: bash

        >>> export SPEY_OPTIMISER=<name>

    spey will automatically track ``SPEY_OPTIMISER`` settings.

    .. versionadded:: 0.2.6

    Args:
        name (``str``): name of the optimiser, ``scipy`` or ``minuit``.
    """
    log.debug(
        "Currently optimiser is set to: `%s`", os.environ.get("SPEY_OPTIMISER", "scipy")
    )
    if name in ["minuit", "iminuit"]:
        if find_spec("iminuit") is not None:
            os.environ["SPEY_OPTIMISER"] = "minuit"
            log.debug("Optimiser set to minuit.")
        else:
            log.error("iminuit package is not available.")
    elif name == "scipy":
        os.environ["SPEY_OPTIMISER"] = "scipy"
        log.debug("Optimiser set to scipy.")
    else:
        log.error(
            "Unknown optimiser: %s. The optimiser is set to %s",
            name,
            os.environ.get("SPEY_OPTIMISER", "scipy"),
        )


if int(os.environ.get("SPEY_LOGLEVEL", -1)) >= 0:
    set_log_level(int(os.environ.get("SPEY_LOGLEVEL")))

if os.environ.get("SPEY_CHECKUPDATE", "ON").upper() != "OFF":
    check_updates()
