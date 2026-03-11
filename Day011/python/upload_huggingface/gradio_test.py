import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from transformers import pipeline
from modelscope.utils.constant import Tasks
import gradio

import warnings
warnings.filterwarnings("ignore")
import os

_model_path = "cache/storier"
# _model_path = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"

# _model = AutoModelForCausalLM.from_pretrained(_model_path,  device_map="cuda",trust_remote_code=False)
# print(_model)
# exit()

_tokenizer = AutoTokenizer.from_pretrained(_model_path,trust_remote_code=True)

print(_tokenizer.tokenize("<s>我爱北京天安门。</s><s>天安门上太阳升</s>"))
print(_tokenizer("<s>我爱北京天安门。</s><s>天安门上太阳升</s>",add_special_tokens=True))

# exit()
# print(_tokenizer.chat_template)

# exit()
# _pp = pipeline(task=Tasks.text_generation,
#                model=_model,
#                tokenizer=_tokenizer,
#                trust_remote_code=True)
# print(_pp("我是中国"))
# gradio.Interface.from_pipeline(_pp).launch(server_name="0.0.0.0",server_port=60001, share=True)

# prompt = "你是谁"
# message = [
#     {"role": "user", "content": prompt}
# ]
#
# text = _tokenizer.apply_chat_template(
#     message,
#     tokenize=False,
#     add_generation_prompt=True
# )
# prompt = "星星为什么会一闪一闪？"
# model_inputs = _tokenizer([f"<s>user\n{prompt}<s></s>assitant\n"], return_tensors="pt").to("cuda")
# # print(model_inputs)
#
# _generation_config = GenerationConfig(
#         do_sample=True,
#         temperature=0.3,
#         top_k=5
#     )
#
# generated_ids = _model.generate(
#     model_inputs.input_ids,
#     generation_config = _generation_config
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
#
# response = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(prompt+response)