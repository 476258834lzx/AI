# import os
import warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

warnings.filterwarnings("ignore")

from transformers import (pipeline,
                          AutoModelForCausalLM,
                          AutoTokenizer)
from peft import PeftModel

_tokenizer_id = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
# _model_id = "/root/workspace/PEFT/myqwen2.5-0.5b"

_model = AutoModelForCausalLM.from_pretrained(_tokenizer_id)
_tokenizer = AutoTokenizer.from_pretrained(_tokenizer_id)
peft_model = PeftModel.from_pretrained(model=_model, model_id="checkpoints/qlora/checkpoint-300")
peft_model = peft_model.merge_and_unload()
# peft_model.save_pretrained("myqwen2.5-0.5b")

pipe = pipeline("text-generation", model=peft_model,
                tokenizer=_tokenizer, device="cuda:0")
ipt = f"User: 你是谁？Assistant:"
print(pipe(ipt))
