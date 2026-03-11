from typing import Union
from .model_config import ModelConfig


def resolve_parameter_index(parameter: Union[int, str], cfg: ModelConfig) -> int:
    r"""
    Validate and convert a nuisance-parameter identifier to its integer index.

    Args:
        parameter (``int`` or ``str``): Parameter index or name.
        cfg (:obj:`~spey.base.model_config.ModelConfig`): Model configuration.

    Raises:
        :obj:`ValueError`: If the model has fewer than 2 parameters, if the index
            is out of range, if the index refers to the POI, or if the name is not
            found in :attr:`~spey.base.model_config.ModelConfig.parameter_names`.

    Returns:
        ``int``: Resolved parameter index.
    """
    if cfg.npar < 2:
        raise ValueError(
            "Nuisance-parameter profiling requires at least 2 model parameters "
            f"(POI + at least one nuisance), but this model has only {cfg.npar}."
        )
    if isinstance(parameter, str):
        if cfg.parameter_names is None:
            raise ValueError(
                "Cannot resolve parameter name: parameter_names is not set in ModelConfig."
            )
        if parameter not in cfg.parameter_names:
            raise ValueError(
                f"Parameter '{parameter}' not found in parameter_names: {cfg.parameter_names}"
            )
        return cfg.parameter_names.index(parameter)
    param_idx = int(parameter)
    if not 0 <= param_idx < cfg.npar:
        raise ValueError(f"Parameter index {param_idx} is out of range [0, {cfg.npar}).")
    if param_idx == cfg.poi_index:
        raise ValueError(
            f"Parameter index {param_idx} refers to the primary POI. "
            "Leave parameter=None to profile the POI instead."
        )
    return param_idx
