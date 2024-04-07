"""
FunctionCallAccuracy metric, used for evaluating the accuracy of the model's choice of function to call and its parameters.
"""
import json
import statistics
from typing import Dict, List, Optional

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


def extract_json_fn_call(msg: str) -> Optional[Dict]:
    try:
        msg = msg[next(idx for idx, c in enumerate(msg) if c in "{[") :]
    except StopIteration:
        return None
    try:
        return json.loads(msg)
    except json.JSONDecodeError as exc:
        return json.loads(msg[: exc.pos])

    return None


class FunctionCallAccuracy(evaluate.Metric):
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
                    "system_messages": datasets.Value("string"),
                    "function_call_messages": datasets.Value("string"),
                }
            ),
        )

    def _compute(
        self, system_messages: List[str], function_call_messages: List[str]
    ) -> Dict[str, float]:
        fn_name_accuracies, fn_param_accuracies = [], []
        for system_message, function_call_message in zip(
            system_messages, function_call_messages
        ):
            system_fn_call = extract_json_fn_call(system_message)
            execution_fn_call = extract_json_fn_call(function_call_message)

            # If either function call is not available, return 1 accuracy
            if not system_fn_call or not execution_fn_call:
                return {"function_choice_accuracy": 1, "parameter_accuracy": 1}

            fn_name_accuracy = (
                1 if system_fn_call["name"] == execution_fn_call["name"] else 0
            )
            total_params, correct_params = 0, 0
            for key in system_fn_call["parameters"]:
                total_params += 1
                if (
                    system_fn_call["parameters"][key]
                    == execution_fn_call["arguments"][key]
                ):
                    correct_params += 1
            fn_param_accuracy = correct_params / total_params if total_params > 0 else 0

            fn_param_accuracies.append(fn_param_accuracy)
            fn_name_accuracies.append(fn_name_accuracy)

        return {
            "function_choice_accuracy": statistics.mean(fn_name_accuracies),
            "parameter_accuracy": statistics.mean(fn_param_accuracies),
        }
