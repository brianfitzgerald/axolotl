"""Module to load prompt strategies."""

import importlib
import inspect
import logging

from axolotl.prompt_strategies.user_defined import UserDefinedDatasetConfig

LOG = logging.getLogger("axolotl.prompt_strategies")


def load(strategy, tokenizer, cfg, ds_cfg):
    try:
        load_fn = "load"
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(f".{strategy}", "axolotl.prompt_strategies")
        func = getattr(mod, load_fn)
        load_kwargs = {}
        if strategy == "user_defined":
            load_kwargs["ds_cfg"] = UserDefinedDatasetConfig(**ds_cfg)
        else:
            sig = inspect.signature(func)
            if "ds_cfg" in sig.parameters:
                load_kwargs["ds_cfg"] = ds_cfg
        return func(tokenizer, cfg, **load_kwargs)
    except ModuleNotFoundError:
        LOG.error(f"Prompt strategy {strategy} not found.")
        return None
