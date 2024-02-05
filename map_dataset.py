import fire
from datasets import load_dataset
import re

ROLE_DICT = {
    "ASSISTANT": "gpt",
    "USER": "human",
    "SYSTEM": "gpt",
}


def chatml_to_json(conversation: str):
    pattern = r"(SYSTEM|USER|ASSISTANT):"

    conversation = conversation.replace("<|endoftext|>", "").replace("\n", " ")

    conversation_steps = []
    matches = re.finditer(pattern, conversation, re.DOTALL)

    start_index = 0

    for i, m in enumerate(matches):
        role_match = m.group(1)
        role = ROLE_DICT[role_match]
        match_start = m.start()
        match_end = m.end()

        # Split the string from the start_index to the beginning of the match
        part = conversation[start_index:match_start]

        # Update the start_index for the next split
        start_index = match_end
        if i > 0:
            conversation_steps[-1]["value"] = part
        conversation_steps.append({"from": role, "value": ""})

    return conversation_steps


def concatenate_columns(batch):
    """Concatenate 'chat' and 'system' columns into a new 'conversation' column."""

    conversations = []

    for chat, system in zip(batch["chat"], batch["system"]):
        conversation = chat + " " + system
        conversations.append(chatml_to_json(conversation))

    batch["conversations"] = conversations
    return batch


def main(test: bool = False):

    if test:
        print(
            concatenate_columns(
                {
                    "chat": [
                        "USER: I need to create a new contact for my friend John Doe. His email is johndoe@example.com. ASSISTANT: <functioncall> ASSISTANT: I have successfully created a new contact for your friend John Doe with the email johndoe@example.com. <|endoftext|>"
                    ],
                    "system": [
                        "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required"
                    ],
                }
            )
        )
        return

    # Load the dataset
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

    modified_dataset = dataset.map(
        concatenate_columns, batched=True, remove_columns=["chat", "system"]
    )

    dataset_id = "roborovski/glaive-function-calling-v2-conversation"
    modified_dataset.push_to_hub(dataset_id)


if __name__ == "__main__":
    # Replace 'your_dataset_name_here' and 'your_split' with the actual dataset name and split you want to process
    fire.Fire(main)
