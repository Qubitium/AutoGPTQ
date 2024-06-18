from ._base import BaseGPTQForCausalLM


class BaiChuanGPTQ(BaseGPTQForCausalLM):
    # non-layer (root) modules
    non_layer_modules = ["model.embed_tokens", "model.norm"]

    # repeating layers
    layers_node = "model.layers"
    layer_type = "DecoderLayer"
    layer_modules = [
        ["self_attn.W_pack"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
