import torch
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.8,
    top_k=5,

    max_tokens=20,
)

from transformers import Qwen2ForCausalLM

# _model_id = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
_model_id = "./cache/storier_vllm"
llm = LLM(model=_model_id,
          dtype=torch.float32,
          trust_remote_code=True)
# print(llm.get_tokenizer().vocab_size,"**************************")
# exit()
print("____________________________")
outputs = llm.generate(["今天"], sampling_params)
print("*******************************")
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"{prompt}{generated_text}")
