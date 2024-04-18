# pylint: disable=redefined-outer-name
"""
Test classes for checking functionality of the cfg normalization
"""

from pytest import fixture

from axolotl.utils.callbacks.tool_eval import FunctionCallAccuracy, extract_json_fn_call
from axolotl.utils.tokenization import chatml_to_conversation

# tuple of (system, user) chat samples
NO_FN_AVAILABLE = {
    "system": """
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { "name": "get_exchange_rate", "description": "Get the exchange rate between two currencies", "parameters": { "type": "object", "properties": { "base_currency": { "type": "string", "description": "The currency to convert from" }, "target_currency": { "type": "string", "description": "The currency to convert to" } }, "required": [ "base_currency", "target_currency" ] } }
""",
    "chat": """
USER: Can you book a flight for me from New York to London? ASSISTANT: I'm sorry, but I don't have the capability to book flights. My current function allows me to get the exchange rate between two currencies. If you need help with that, feel free to ask! <|endoftext|>
""",
    "expected": """
USER: Can you book a flight for me from New York to London? ASSISTANT: I'm sorry, but I don't have the capability to book flights. My current function allows me to get the exchange rate between two currencies. If you need help with that, feel free to ask! <|endoftext|>
""",
}

FN_AVAILABLE = {
    "system": """
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { "name": "calculate_loan_payment", "description": "Calculate the monthly loan payment", "parameters": { "type": "object", "properties": { "principal": { "type": "number", "description": "The principal amount of the loan" }, "interest_rate": { "type": "number", "description": "The annual interest rate of the loan" }, "loan_term": { "type": "integer", "description": "The loan term in years" } }, "required": [ "principal", "interest_rate", "loan_term" ] } }
""",
    "chat": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: <functioncall> {"name": "calculate_loan_payment", "arguments": '{"principal": 50000, "interest_rate": 5, "loan_term": 4}'}
""",
    "expected": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: <functioncall> {"name": "calculate_loan_payment", "arguments": '{"principal": 50000, "interest_rate": 5, "loan_term": 4}'}
""",
}

WRONG_FN_NAME = {
    "system": """
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { "name": "calculate_loan_payment", "description": "Calculate the monthly loan payment", "parameters": { "type": "object", "properties": { "principal": { "type": "number", "description": "The principal amount of the loan" }, "interest_rate": { "type": "number", "description": "The annual interest rate of the loan" }, "loan_term": { "type": "integer", "description": "The loan term in years" } }, "required": [ "principal", "interest_rate", "loan_term" ] } }
""",
    "chat": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: <functioncall> {"name": "calculate_loan_payment", "arguments": '{"base_amount": 50000, "rate": 5, "loan_term": 4}'}
""",
    "expected": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: <functioncall> {"name": "get_loan", "arguments":  "arguments": '{"base_amount": 50000, "rate": 5, "loan_term": 4}'}
""",
}

WRONG_PARAMETERS = {
    "system": """
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { "name": "calculate_loan_payment", "description": "Calculate the monthly loan payment", "parameters": { "type": "object", "properties": { "principal": { "type": "number", "description": "The principal amount of the loan" }, "interest_rate": { "type": "number", "description": "The annual interest rate of the loan" }, "loan_term": { "type": "integer", "description": "The loan term in years" } }, "required": [ "principal", "interest_rate", "loan_term" ] } }
""",
    "chat": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: <functioncall> {"name": "calculate_loan_payment", "arguments": '{"base_amount": 50000, "rate": 5, "loan_term": 4}'}
""",
    "expected": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: <functioncall> {"name": "get_loan", "arguments": '{"base_amount": 50000, "rate": 5, "loan_term": 4}'}
""",
}


@fixture()
def metric():
    return FunctionCallAccuracy()


def _assert_conversation(metric, conversation, score):
    system_msg = chatml_to_conversation(conversation["system"], "system")[-1]["value"]
    last_msg = chatml_to_conversation(conversation["chat"], "chat")[-1]["value"]
    expected = chatml_to_conversation(conversation["expected"], "chat")[-1]["value"]

    metric.add_batch(
        references=[expected], predictions=[last_msg], sources=[system_msg]
    )

    values = metric.compute()

    assert values, "No values returned"
    assert values["score"] == score


def test_json_parse():
    prompt = FN_AVAILABLE["chat"]
    parsed = extract_json_fn_call(prompt)
    assert isinstance(parsed, dict)


def test_no_fn_available(metric):
    _assert_conversation(metric, NO_FN_AVAILABLE, 0)


def test_fn_available(metric):
    _assert_conversation(metric, FN_AVAILABLE, 1)


def test_wrong_fn(metric):
    _assert_conversation(metric, WRONG_FN_NAME, 0)


def test_wrong_parameters(metric):
    _assert_conversation(metric, WRONG_PARAMETERS, 0)
