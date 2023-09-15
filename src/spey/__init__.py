from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import pkg_resources
from semantic_version import SimpleSpec, Version

from spey.base import BackendBase, ConverterBase
from spey.combiner import UnCorrStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.system.exceptions import PluginError

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
]


def __dir__():
    return __all__


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


def AvailableBackends() -> List[Text]:
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
        }

    raise PluginError(
        f"The backend {name} is unavailable. Available backends are "
        + ", ".join(AvailableBackends())
        + "."
    )
