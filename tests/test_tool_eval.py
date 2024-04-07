# pylint: disable=redefined-outer-name
"""
Test classes for checking functionality of the cfg normalization
"""

from pytest import fixture

from axolotl.utils.callbacks.tool_eval import FunctionCallAccuracy
from axolotl.utils.tokenization import chatml_to_conversation

# tuple of (system, user) chat samples
NO_FN_AVAILABLE = {
    "system": """
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { "name": "get_exchange_rate", "description": "Get the exchange rate between two currencies", "parameters": { "type": "object", "properties": { "base_currency": { "type": "string", "description": "The currency to convert from" }, "target_currency": { "type": "string", "description": "The currency to convert to" } }, "required": [ "base_currency", "target_currency" ] } }
""",
    "chat": """
USER: Can you book a flight for me from New York to London? ASSISTANT: I'm sorry, but I don't have the capability to book flights. My current function allows me to get the exchange rate between two currencies. If you need help with that, feel free to ask! <|endoftext|>
""",
}
FN_AVAILABLE = {
    "system": """
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { "name": "calculate_loan_payment", "description": "Calculate the monthly loan payment", "parameters": { "type": "object", "properties": { "principal": { "type": "number", "description": "The principal amount of the loan" }, "interest_rate": { "type": "number", "description": "The annual interest rate of the loan" }, "loan_term": { "type": "integer", "description": "The loan term in years" } }, "required": [ "principal", "interest_rate", "loan_term" ] } }
""",
    "chat": """
USER: Hi, I need help with calculating my loan payment. ASSISTANT: Of course, I can help with that. Could you please provide me with the principal amount of the loan, the annual interest rate, and the loan term in years? <|endoftext|> USER: Sure, the principal amount is $50000, the annual interest rate is 5%, and the loan term is 10 years. ASSISTANT: <functioncall> {"name": "calculate_loan_payment", "arguments": '{"principal": 50000, "interest_rate": 5, "loan_term": 10}'} <|endoftext|> FUNCTION RESPONSE: {"monthly_payment": "$530.33"} ASSISTANT: Based on the information you provided, your monthly loan payment would be $530.33. <|endoftext|>
""",
}


@fixture()
def metric():
    return FunctionCallAccuracy()


def _assert_conversation(metric, conversation):
    conversation = chatml_to_conversation(NO_FN_AVAILABLE)
    system_msg = conversation[0]["value"]
    last_msg = conversation[-1]["value"]

    metric.add_batch(system_messages=[system_msg], function_call_messages=[last_msg])

    values = metric.compute()

    assert values, "No values returned"
    assert values["function_choice_accuracy"] == 1
    assert values["parameter_accuracy"] == 1


def test_no_fn_available(metric):
    _assert_conversation(metric, NO_FN_AVAILABLE)


def test_fn_available(metric):
    _assert_conversation(metric, FN_AVAILABLE)
