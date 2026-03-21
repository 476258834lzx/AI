from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset


# _model_path = "/root/workspace/skyer_huggingface/cache/storier"
_model_path ="/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=False)
# _config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
# _config.cache_max_batch_size = None
_model = AutoModelForCausalLM.from_pretrained(_model_path,
                                            #   config=_config,
                                              trust_remote_code=False)


_dataset = load_dataset("json", data_files="ruozhiba_qa.json", split="train")


def preprocess_dataset(data):
    return {"text": f"<s>user\n{data['instruction']}</s><s>assistant\n{data['output']}</s>"}


_dataset = _dataset.map(preprocess_dataset, remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()

_response_template = "<s>assistant\n"
_collator = DataCollatorForCompletionOnlyLM(_response_template, tokenizer=_tokenizer)

_training_args = SFTConfig(
    output_dir="checkpoints/sft",
    dataset_text_field="text",
    max_seq_length=512,
    per_device_train_batch_size=5,
    gradient_accumulation_steps=1,
    num_train_epochs=6,
    save_steps=100,
    logging_steps=10
    # optim="paged_adamw_32bit"
)

_trainer = SFTTrainer(
    model=_model,
    tokenizer=_tokenizer,
    args=_training_args,
    train_dataset=_dataset,
    data_collator=_collator
)

_trainer.train()


#新版移除DataCollatorForCompletionOnlyLM

# from transformers import (AutoModelForCausalLM,
#                           AutoTokenizer,
#                           AutoConfig)
# from trl import SFTConfig, SFTTrainer
# from datasets import load_dataset
#
# # _model_path = "./cache/skyer"
# _model_path = "/root/Workspace/airelearn/Day011/python/TRL/cache/Qwen/Qwen2___5-0___5B-Instruct"
# _tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=False)
# # _config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
# # _config.cache_max_batch_size = None
# _model = AutoModelForCausalLM.from_pretrained(_model_path,
#                                               #   config=_config,
#                                               trust_remote_code=False)
#
# _dataset = load_dataset("json", data_files="ruozhiba_qa.json", split="train")
#
#
# def preprocess_dataset(data):
#     return {"text": f"<s>user\n{data['instruction']}</s><s>assistant\n{data['output']}</s>"}
#
#
# _dataset = _dataset.map(preprocess_dataset, remove_columns=_dataset.column_names)
# _dataset = _dataset.shuffle()
#
# # _response_template = "<s>assistant\n"
# # _collator = DataCollatorForCompletionOnlyLM(_response_template, tokenizer=_tokenizer)
#
# _training_args = SFTConfig(
#     output_dir="checkpoints/sft",
#     dataset_text_field="text",
#
#     per_device_train_batch_size=5,
#     gradient_accumulation_steps=1,
#     num_train_epochs=6,
#     save_steps=100,
#     logging_steps=10
#     # optim="paged_adamw_32bit"
# )
#
# _trainer = SFTTrainer(
#     model=_model,
#     processing_class=_tokenizer,
#     args=_training_args,
#     train_dataset=_dataset,
#
# )
#
# _trainer.train()
