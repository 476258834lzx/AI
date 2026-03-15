from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset
# from modelscope import MsDataset


_model_path = "./cache/storier"
# _model_path = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
_config.cache_max_batch_size = None
_model = AutoModelForCausalLM.from_pretrained(_model_path,
                                              config=_config,
                                              trust_remote_code=True)

# import torch
# a =  torch.tensor([2,  1084, 25164,  1746,  1851, 25220,  7042,  5513, 25189,     3,
#             2,  5254, 25164,   871, 25408, 25388, 26010,  1026, 25795, 26932,
#         25945, 25288,     3,     2, 11789,   555,  1343,  1343, 25176,  2025,
#         25529, 25408,   824, 25176, 25263, 25795,  5092,  4933, 16054,  3769,
#         27104, 25408, 25388, 27109, 25185, 25176, 25576,  2019,  5092, 26987,
#         25185,  9443, 25433, 25433,  4019, 25252, 14279,  2033, 25506,  1237,
#         25263, 14810, 25176, 25366,  1781,  5513, 22389, 25203, 25395,  3571,
#         25395,   939,  1619, 25566, 26027, 25189,  1486, 13917,  6728,  1237,
#         25176,  7610,  1225, 25529, 13584, 25176, 15876, 25506, 25366, 25209,
#         25189,   318,  2372, 18718, 25189, 25164,  2127, 25176, 25331,  7683,
#         25263, 25789, 14366,  1743, 25942, 25344,  9773,  9773, 18402,  3211,
#         25189, 25164,   173, 25554, 25460, 25335,  3571, 25263, 19849, 25185,
#         26382, 25461, 25203, 19904, 26859, 26107, 25220,   461, 26382, 25289,
#         25288, 25442,  3976, 25493,  8340, 25288, 25164, 25408, 25579, 26932,
#         25945,  1851, 25176, 25795,   820,  3437, 25510, 26950, 25384, 25176,
#         25263,  3338, 25366, 25263,  4737, 25795, 26932, 25945, 25185,  1352,
#         25189,  1316, 25366,  1206, 23990, 25326, 25327, 25176,  1983,  1116,
#         24826, 25176, 25472,  5105, 22373, 25366, 26877, 25435,  8369, 25413,
#          4962, 26508, 27300, 25185,  9226, 25428, 26932, 25945, 25189,  7191,
#         25366, 25204, 28967, 28932, 13917, 14905, 15176, 25289, 11047, 25925,
#         25189, 25789,   569, 25263,  2085, 25206,   871, 25911, 26685, 25683,
#         25185,  4278, 25211, 25176, 26932, 25945, 14500, 25254, 24826,  3864,
#         25642, 24189, 25189, 25164,   609,  1825, 25395,   658,   173,  3338,
#         25366, 26932, 25945,  4126,  8588, 26932, 25359,   867, 22832,  7220,
#         15791,  2392, 25204, 20303, 25911, 26187, 25185, 12744,  1825, 26932,
#         25945,  8640,   173, 26932,  4437, 25185, 22832, 25176,  4151,   961,
#         25252,  1976,   459,  1185, 25189, 12211, 18786,  4839, 25220, 26932,
#         25945,  4887, 15237,   406, 26932, 26731, 25646, 25222, 26932, 26452,
#         27783, 25176, 25263, 14810, 25206,   871, 25911, 20783, 25222, 26452,
#         27783, 25189,   492, 26932, 25945,   569,  5092, 25566, 21145, 25176,
#         25580,  4000, 25209,  5639, 25176,  5656, 24467, 24297,  1681, 25164,
#         25234,  6704, 25412, 25202, 25176, 26932, 25945,  2292, 25235,  5141,
#          4518,   469,  7013,  4516, 25176,  3961,  3689, 13787, 25176, 21962,
#          1743, 27588, 25496, 26778,  1329, 25251,   820, 25176, 26932, 25945,
#          1225,  1686,  1653,   229, 29512, 25780,  2494, 25185,  7000, 25213,
#         26107, 25189, 25205, 14878,   791,   360, 26144, 26859, 25185, 26107,
#         28661, 25185, 25558, 26310, 25517,   670,  1237, 12278, 25189, 25164,
#          3759,  1095,  8518,  3339, 25203, 25164, 25192, 25193,  3930, 25254,
#         17406, 25473, 26549, 25260, 25176,   815,  2784, 25204, 15216, 15052,
#         25176, 25164, 10876, 25366, 25220, 25176, 13903, 25911, 25286,  5786,
#         25189, 25164, 25194, 25193, 25247, 26989, 26022, 25362, 25412, 25176,
#         25749, 25585, 25450, 25362, 25471, 25189,   379,   464,  2316,   871,
#         25235, 25911, 25256,  2911,  4516, 25189, 25164, 25210, 25193,  9619,
#         25263,  1116, 25176,  1326, 21751, 25189, 25263, 25358,  1009, 11089,
#          7246, 26027, 25288,     3])
# print(_tokenizer.decode(a))


# print(_model)
# for k,v in _model.named_parameter():
#     if k.endswith("bias"):
#         v.require_grad = False

# exit()
_dataset = load_dataset("json", data_files="./ruozhiba_qa.json", split="train")

def preprocess_dataset(example):
    MAX_LENGTH = 128
    _input_ids, _attention_mask, _labels = [], [], []
    _instruction = _tokenizer(f"<s>user\n{example['instruction']}</s>\n<s>assistant\n", add_special_tokens=False)
    _response = _tokenizer(example["output"] + _tokenizer.eos_token, add_special_tokens=False)

    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + _response["attention_mask"]

    _labels = [-100] * len(_instruction["input_ids"]) + _response["input_ids"]#hugging face 定义-100不参与训练

    if len(_input_ids) > MAX_LENGTH:#手动截断
        _input_ids = _input_ids[:MAX_LENGTH]
        _attention_mask = _attention_mask[:MAX_LENGTH]
        _labels = _labels[:MAX_LENGTH]

    return {
        "input_ids": _input_ids,
        "attention_mask": _attention_mask,
        "labels": _labels
    }#返回的键必须是这三个，否则hugging face不认


_dataset = _dataset.map(preprocess_dataset, remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()

_training_args = TrainingArguments(
    output_dir="checkpoints/train",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_steps=100,
    logging_steps=100,
    # deepspeed="deepspeed_config.json"
    # optim="paged_adamw_32bit",
)


trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer, padding=True),
)
trainer.train()


# if __name__ == '__main__':

#     _model = AutoModelForCausalLM.from_pretrained("/root/workspace/PEFT/checkpoints/train/checkpoint-300")

#     _tokenizer = AutoTokenizer.from_pretrained(_model_path)

#     from transformers import pipeline

#     pp= pipeline(task = "text-generation",model=_model,tokenizer=_tokenizer)
#     print(pp("你好"))
