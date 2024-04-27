from typing import Dict, List, Optional

import datasets
import evaluate
import torch
from tqdm import tqdm

_KWARGS_DESCRIPTION = ""


class Perplexity:
    """
    Calculate perplexity as defined in https://huggingface.co/docs/transformers/en/perplexity.
    This is a custom variant that doesn't re-tokenize the input or re-load the model.
    """

    def __init__(self, max_seq_len: int, stride: int = 512) -> None:
        self.max_seq_len = max_seq_len
        self.stride = stride

    def compute(
        self,
        input_ids: List[int],
        labels: List[int],
        predictions: List[int],
    ) -> Dict[str, float]:
        stride = 512
        input_ids_t = torch.tensor(input_ids)
        labels_t = torch.tensor(labels)
        predictions_t = torch.tensor(predictions)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids_t = input_ids_t[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
