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

_tokenizer = AutoTokenizer.from_pretrained(_model_path,trust_remote_code=True)
_model = AutoModelForCausalLM.from_pretrained(_model_path,  device_map="cuda",trust_remote_code=True)

print(_tokenizer.tokenize("<s>我爱北京天安门。</s><s>天安门上太阳升</s>"))
print(_tokenizer("我爱北京天安门。天安门上太阳升",max_length=5,padding="max_length",truncation=True,add_special_tokens=True))
print(_tokenizer("<s>我爱北京天安门。</s><s>天安门上太阳升</s>",max_length=5,padding="max_length",truncation=True,add_special_tokens=True))
print(_tokenizer.encode("<s>我爱北京天安门。</s><s>天安门上太阳升</s>",add_special_tokens=False))
print(_tokenizer.decode([2, 5, 26895, 475, 36903, 1, 3, 2, 5, 36903, 17, 2820, 1550, 3],skip_special_tokens=True))
print(_tokenizer.vocab_size)
print(_tokenizer.vocab_files_names)

# print(_tokenizer.chat_template)

# 推理
# _pp = pipeline(task=Tasks.text_generation,
#                model=_model,
#                tokenizer=_tokenizer,
#                trust_remote_code=True)
# print(_pp("我是中国"))
# gradio.Interface.from_pipeline(_pp).launch(server_name="0.0.0.0",server_port=60001, share=True)

prompt = "你是谁"
message = [
    {"role": "user", "content": prompt}
]

text = _tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)
prompt = "星星为什么会一闪一闪？"
model_inputs = _tokenizer([f"<s>user\n{prompt}<s></s>assitant\n"], return_tensors="pt").to("cuda")
print(model_inputs)
#
_generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.3,
        top_k=5
    )

generated_ids = _model.generate(
    model_inputs.input_ids,
    generation_config = _generation_config
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(prompt+response)


# from transformers import AutoModel, AutoConfig,AutoModelForCausalLM
# # 打印当前工作目录
# import os
# print("Current working directory:", os.getcwd())
#
# # 使用 AutoConfig 加载自定义的配置
# model_dir = "./cache/storier"
# config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
#
# # 使用 AutoModel 加载自定义的模型
# model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True)
#
# # 打印模型配置
# print("Model configuration:", config)
#
# # 打印模型架构
# print("Model architecture:", model)