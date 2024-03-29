from ._base import BaseGPTQForCausalLM


class DbrxGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DbrxBlock"
    layers_block_name = "transformer.blocks"
    outside_layer_modules = ["transformer.wte", "transformer.norm_f"]
    inside_layer_modules = [
        ["norm_attn_norm.norm_1"],
        ["norm_attn_norm.norm_2"],
        ["norm_attn_norm.attn.Wqkv"],
        ["norm_attn_norm.attn.out_proj"],
        ["ffn.router.layer"],
        ["ffn.experts.mlp.0.w1", "ffn.experts.mlp.0.v1"], ["ffn.experts.mlp.0.w2"],
        ["ffn.experts.mlp.1.w1", "ffn.experts.mlp.1.v1"], ["ffn.experts.mlp.1.w2"],
        ["ffn.experts.mlp.2.w1", "ffn.experts.mlp.2.v1"], ["ffn.experts.mlp.2.w2"],
        ["ffn.experts.mlp.3.w1", "ffn.experts.mlp.3.v1"], ["ffn.experts.mlp.3.w2"],
        ["ffn.experts.mlp.4.w1", "ffn.experts.mlp.4.v1"], ["ffn.experts.mlp.4.w2"],
        ["ffn.experts.mlp.5.w1", "ffn.experts.mlp.5.v1"], ["ffn.experts.mlp.5.w2"],
        ["ffn.experts.mlp.6.w1", "ffn.experts.mlp.6.v1"], ["ffn.experts.mlp.6.w2"],
        ["ffn.experts.mlp.7.w1", "ffn.experts.mlp.7.v1"], ["ffn.experts.mlp.7.w2"],
        ["ffn.experts.mlp.8.w1", "ffn.experts.mlp.8.v1"], ["ffn.experts.mlp.8.w2"],
        ["ffn.experts.mlp.9.w1", "ffn.experts.mlp.9.v1"], ["ffn.experts.mlp.9.w2"],
        ["ffn.experts.mlp.10.w1", "ffn.experts.mlp.10.v1"], ["ffn.experts.mlp.10.w2"],
        ["ffn.experts.mlp.11.w1", "ffn.experts.mlp.11.v1"], ["ffn.experts.mlp.11.w2"],
        ["ffn.experts.mlp.12.w1", "ffn.experts.mlp.12.v1"], ["ffn.experts.mlp.12.w2"],
        ["ffn.experts.mlp.13.w1", "ffn.experts.mlp.13.v1"], ["ffn.experts.mlp.13.w2"],
        ["ffn.experts.mlp.14.w1", "ffn.experts.mlp.14.v1"], ["ffn.experts.mlp.14.w2"],
        ["ffn.experts.mlp.15.w1", "ffn.experts.mlp.15.v1"], ["ffn.experts.mlp.15.w2"],
    ]


__all__ = ["DbrxGPTQForCausalLM"]
