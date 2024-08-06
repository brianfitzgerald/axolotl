from typing import List, Dict, Optional


ENTITY_EXTRACTION_TUNING_INSTRUCTION = (
    "Extract structured data from the following context in JSON form."
)


def process_entity_extracton(sample: dict) -> List[Dict[str, str]]:

    # these fields are misnamed in the dataset
    sample_query = sample["json_data"]
    sample_data = sample["json_query"]

    completion_msg = f"```json\n{sample_data}\n```"

    conversation = [
        {
            "role": "user",
            "content": sample["context"],
        },
        {
            "role": "user",
            "content": sample_query,
        },
        {
            "role": "assistant",
            "content": completion_msg,
        },
    ]
    return conversation


def process_goody(sample: dict) -> List[Dict[str, str]]:

    conversation = [
        {
            "role": "user",
            "content": sample["instruction"],
        },
        {
            "role": "assistant",
            "content": sample["response"],
        },
    ]
    return conversation


def get_preprocessor(
    processor_name: Optional[str], sample: dict
) -> Optional[List[Dict[str, str]]]:
    if processor_name:
        if processor_name == "entity_extraction":
            return process_entity_extracton(sample)
        if processor_name == "goody":
            return process_goody(sample)
    return None
