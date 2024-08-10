from typing import Optional
from transformers import AutoConfig
import fire
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str  # model config name
    num_layers: int  # number of transformer layers (blocks)
    n_head: int  # number of attention heads
    hidden_dim: int  # hidden dimension
    vocab_size: int  # vocabulary size
    max_seq_len: int  # max sequence length
    num_key_value_heads: int  # the number of key value heads implementing Grouped Query Attention (GQA), If it is not specified, will default to n_head. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. See https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py for details
    num_key_value_groups: int  # number of key value groups for GQA
    ffn_embed_dim: int

    def __post_init__(self):
        if self.ffn_embed_dim is None and self.expansion_ratio is None:
            self.ffn_embed_dim = self.hidden_dim * 4
            self.expansion_ratio = 4
        elif self.ffn_embed_dim is None:
            self.ffn_embed_dim = self.hidden_dim * self.expansion_ratio
        elif self.expansion_ratio is None:
            self.expansion_ratio = self.ffn_embed_dim / self.hidden_dim

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head
        assert self.n_head % self.num_key_value_heads == 0, f"n_head ({self.n_head}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        self.num_key_value_groups = self.n_head / self.num_key_value_heads
    

def params_per_attn_layer(config: ModelConfig):

    n_kv_heads = config.num_key_value_heads

    hidden_dim_params = 2 * config.hidden_dim ** 2
    kv_params = 2 * config.hidden_dim * (config.hidden_dim * n_kv_heads / config.n_head)
    return hidden_dim_params + kv_params

def params_per_mlp_layer(config: ModelConfig):

    return 3 * config.hidden_dim * config.ffn_embed_dim

def main(model_name: str):
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)


if __name__ == "__main__":
    fire.Fire(main)
