"""
DPO strategies for zephyr
"""


def nectar(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        data = {}
        data["prompt"] = (
            "<|system|>\n</s>\n"
            "<|user|>\n"
            f"{sample['prompt']}</s>\n"
            "<|assistant|>\n"
        )
        answers = sorted(sample["answers"], key=lambda x: x["rank"])
        data["chosen"] = answers[-1]["answer"]
        data["rejected"] = answers[-2]["answer"]

        return data

    return transform_fn


def toolformer(
    cfg, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def format_option(tool_call, call_result, agent_output):
        return f"<|user|>{tool_call} {call_result} {agent_output}</>"

    def transform_fn(sample):
        data = {}
        data["prompt"] = (
            "<|system|>\n</s>\n"
            "<|user|>\n"
            f"{sample['prompt']}</s>\n"
            "<|assistant|>\n"
        )
        data["chosen"] = format_option(
            sample["tool_call_accepted"],
            sample["call_result_accepted"],
            sample["agent_output_accepted"],
        )
        data["rejected"] = format_option(
            sample["tool_call_rejected"],
            sample["call_result_rejected"],
            sample["agent_output_rejected"],
        )

        return data

    return transform_fn
