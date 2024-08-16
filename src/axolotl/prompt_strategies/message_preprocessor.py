from typing import List, Dict, Optional, Tuple


ENTITY_EXTRACTION_TUNING_INSTRUCTION = (
    "Extract structured data from the following context in JSON form."
)

# tuple of (conversation, completion)
PreprocessOutput = List[Dict[str, str]]


def markdown_json(json_str: str) -> str:
    return f"```json\n{json_str}\n```"


def process_entity_extraction(sample: dict) -> Tuple[PreprocessOutput, Dict[str, str]]:

    # these fields are misnamed in the dataset
    sample_query = sample["json_data"]
    sample_data = sample["json_query"]
    context = sample["context"]

    conversation = [
        {
            "role": "user",
            "content": f"<context>{context}</context><query>{markdown_json(sample_query)}</query>",
        },
    ]
    return (
        conversation,
        {
            "role": "assistant",
            "content": markdown_json(sample_data),
        },
    )


def process_goody(
    sample: dict, response_field: str
) -> Tuple[PreprocessOutput, Dict[str, str]]:

    conversation = [
        {
            "role": "user",
            "content": sample["input"],
        },
    ]
    return (
        conversation,
        {
            "role": "assistant",
            "content": sample[response_field],
        },
    )


def get_preprocessor(
    processor_name: Optional[str], sample: dict
) -> Optional[PreprocessOutput]:
    if not processor_name:
        return None
    conversation, completion = None, None
    if processor_name == "entity_extraction":
        conversation, completion = process_entity_extraction(sample)
    elif processor_name == "goody":
        conversation, completion = process_goody(sample, "response")
    else:
        raise ValueError(f"Unknown processor name: {processor_name}")
    return conversation + [completion]


def get_dpo_preprocessor(
    processor_name: Optional[str], sample: dict
) -> Optional[Tuple[PreprocessOutput, Dict[str, str], Dict[str, str]]]:
    if processor_name == "goody":
        conv, chosen = process_goody(sample, "chosen")
        _, rejected = process_goody(sample, "rejected")
        return conv, chosen, rejected
    return None
