"""
FunctionCallAccuracy metric, used for evaluating the accuracy of the model's choice of function to call and its parameters.
"""

import statistics
from typing import Dict, List
from datasets import Dataset

import datasets
import evaluate

_KWARGS_DESCRIPTION = """
Args:
    system_message (`str`): The system message containing the available functions.
    function_call_message (`str`): The function call message made by the model.
Returns:
    function_choice_accuracy (`float`): Accuracy of the model's choice of function to call.
    parameter_accuracy (`float`): Accuracy of the model's choice of function parameters.
"""


class CodeExecutionEval(evaluate.Metric):
    """
    Get the accuracy of the model's choice of function to call and its parameters.
    """

    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "references": datasets.Value("string"),
                    "predictions": datasets.Value("string"),
                    "sources": datasets.Value("string"),
                }
            ),
        )

    def _compute(
        self, references: List[str], predictions: List[str], dataset: Dataset
    ) -> Dict[str, float]:

        return {"score": 0}
