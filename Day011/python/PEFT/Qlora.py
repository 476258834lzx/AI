
from peft import (LoraConfig,
                  get_peft_model,
                  TaskType)
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                        AutoTokenizer,
                        AutoConfig,
                        TrainingArguments,
                        Trainer,
                        DataCollatorForSeq2Seq,
                        BitsAndBytesConfig)

_model_id = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
# _model_id = "./cache/storier"

_tokenizer = AutoTokenizer.from_pretrained(_model_id, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_id,trust_remote_code = True)
_config.cache_max_batch_size = None


_bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)

_model = AutoModelForCausalLM.from_pretrained(_model_id,
                                              config =_config,
                                              #   low_cpu_mem_usage=True,
                                              quantization_config=_bnb_config,
                                              trust_remote_code = True)


_dataset = load_dataset("json", data_files="ruozhiba_qa.json", split="train")


def preprocess_dataset(example):
    MAX_LENGTH = 128
    _input_ids, _attention_mask, _labels = [], [], []
    _instruction = _tokenizer(f"<s>user\n{example['instruction']}</s>\n<s>>assistant\n", add_special_tokens=False)
    _response = _tokenizer(example["output"] + _tokenizer.eos_token, add_special_tokens=False)

    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + _response["attention_mask"]

    _labels = [_tokenizer.pad_token_id] * len(_instruction["input_ids"]) + _response["input_ids"]

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
                    r=32,
                    lora_alpha=16,
                    target_modules="all-linear")


_model = get_peft_model(_model, config)
_model.print_trainable_parameters()
_model.enable_input_require_grads()

_training_args = TrainingArguments(
    learning_rate=0.0001,
    output_dir="checkpoints/qlora",
    run_name="qlora_study",
    per_device_train_batch_size=2,
    num_train_epochs=200,
    save_steps=10,
    # deepspeed="deepspeed_config.json"
    optim="paged_adamw_32bit",

)
trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer, padding=True,label_pad_token_id=_tokenizer.pad_token_id),
)

trainer.train()
