"""
FunctionCallAccuracy metric, used for evaluating the accuracy of the model's choice of function to call and its parameters.
"""

import json
import re
import statistics
from typing import Dict, List, Optional, Union

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


JSON_MATCH_PATTERN = r"{.*}"


def recursive_json_parse(data: str) -> Optional[Union[Dict, str]]:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if isinstance(data, dict):
        return {key: recursive_json_parse(value) for key, value in data.items()}
    return data


def extract_json_fn_call(msg: str) -> Optional[Dict]:
    msg = msg.replace("'", "").replace("\n", "")
    match = re.search(JSON_MATCH_PATTERN, msg)

    if match:
        json_str = match.group(0)
        json_obj = recursive_json_parse(json_str)
        return json_obj  # type: ignore
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
                    "references": datasets.Value("string"),
                    "predictions": datasets.Value("string"),
                }
            ),
        )

    def _compute(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        fn_name_accuracies, fn_param_accuracies = [], []
        for expected_message, generated_message in zip(references, predictions):
            system_fn_call = {}
            expected_fn_call = extract_json_fn_call(expected_message)
            execution_fn_call = extract_json_fn_call(generated_message)

            validate = True

            # If a function call is expected but missing, return 0 accuracy
            if bool(system_fn_call) != bool(execution_fn_call):
                validate = False

            if isinstance(execution_fn_call, str) or isinstance(expected_fn_call, str):
                validate = False

            # if we don't have fn calls for all 3 messages, skip
            if (
                not validate
                or not system_fn_call
                or not execution_fn_call
                or not expected_fn_call
            ):
                fn_param_accuracies.append(0.0)
                fn_name_accuracies.append(0.0)
                continue

            fn_name_accuracy = (
                1 if expected_fn_call["name"] == execution_fn_call["name"] else 0
            )

            system_fn_call_arguments = system_fn_call["parameters"]["properties"]
            expected_args = expected_fn_call["arguments"]
            execution_args = execution_fn_call["arguments"]

            total_params, correct_params = 0, 0
            # TODO validate that required params are present
            # TODO validate type as well
            for key in system_fn_call_arguments:
                total_params += 1
                if (
                    key in system_fn_call_arguments
                    and key in expected_args
                    and key in execution_args
                    and expected_args[key] == execution_args[key]
                ):
                    correct_params += 1
            fn_param_accuracy = correct_params / total_params if total_params > 0 else 0

            fn_param_accuracies.append(fn_param_accuracy)
            fn_name_accuracies.append(fn_name_accuracy)

        fn_choice_accuracy = statistics.mean(fn_name_accuracies)
        parameter_accuracy = statistics.mean(fn_param_accuracies)
        return {
            "score": (fn_choice_accuracy + parameter_accuracy) / 2,
        }
