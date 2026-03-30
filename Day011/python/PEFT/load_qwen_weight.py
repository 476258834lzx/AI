import safetensors.torch

weights = safetensors.torch.load_file("model.safetensors")
new_weights = {}
for k, v in weights.items():
    new_k = k.replace("tf_layer._layers.", "layers.") \
              .replace("._att_layer._qw", ".self_attn.qkv_proj") \
              .replace("._att_layer._kw", ".self_attn.qkv_proj") \
              .replace("._att_layer._vw", ".self_attn.qkv_proj") \
              .replace("._att_layer._ow", ".self_attn.o_proj") \
              .replace("._ffn_layer._w0", ".mlp.gate_up_proj") \
              .replace("._ffn_layer._w1", ".mlp.gate_up_proj") \
              .replace("._ffn_layer._w2", ".mlp.down_proj") \
              .replace("._att_norm._w", ".input_layernorm.weight") \
              .replace("._ffn_norm._w", ".post_attention_layernorm.weight") \
              .replace("._tf_layer._out_norm._w", ".norm.weight") \
              .replace("emb.weight", "embed_tokens.weight")
    new_weights[new_k] = v
safetensors.torch.save_file(new_weights, "model.safetensors")