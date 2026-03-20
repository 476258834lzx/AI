from peft import (LoraConfig,
                  get_peft_model,
                  TaskType)
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoConfig,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForSeq2Seq)


# _model_id = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
_model_id = "./cache/storier"

_tokenizer = AutoTokenizer.from_pretrained(_model_id, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_id,trust_remote_code = True)
_config.cache_max_batch_size = None
_model = AutoModelForCausalLM.from_pretrained(_model_id,config=_config, trust_remote_code=True)

# for name ,param in _model.named_parameters():
#     print(name)

_dataset = load_dataset("json", data_files="ruozhiba_qa.json", split="train")


def preprocess_dataset(example):
    MAX_LENGTH = 128
    _input_ids, _attention_mask, _labels = [], [], []
    _instruction = _tokenizer(f"<s>user\n{example['instruction']}</s><s>assistant\n", add_special_tokens=False)
    _response = _tokenizer(example["output"] + _tokenizer.eos_token, add_special_tokens=False)

    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + _response["attention_mask"]

    _labels = [-100] * len(_instruction["input_ids"]) + _response["input_ids"]

    if len(_input_ids) > MAX_LENGTH:
        _input_ids = _input_ids[:MAX_LENGTH]
        _attention_mask = _attention_mask[:MAX_LENGTH]
        _labels = _labels[:MAX_LENGTH]

    return {
        "input_ids": _input_ids,
        "attention_mask": _attention_mask,
        "labels": _labels
    }


_dataset = _dataset.map(preprocess_dataset, remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()


config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    target_modules="all-linear",
                    r=8,
                    lora_alpha=16)


_model = get_peft_model(_model, config)

_model.print_trainable_parameters()


_training_args = TrainingArguments(
    output_dir="checkpoints/lora",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=2,
    # gradient_checkpointing=True,
    num_train_epochs=1,
    save_steps=300
)

trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer, padding=True)
)

trainer.train()
