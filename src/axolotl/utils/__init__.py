"""
Basic utils for Axolotl
"""
import importlib
from typing import List, Optional
import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None

class StopOnTokens(StoppingCriteria):

    def __init__(self, stop_ids: Optional[List[int]] = None):
        super().__init__()
        self.stop_ids = stop_ids or [29, 0]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

