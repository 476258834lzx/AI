import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="sample.txt",
    model_prefix="tokenizer",   # 输出文件前缀
    vocab_size=50000,              # 词表大小
    model_type="unigram",         # 关键：指定 Unigram 算法
    character_coverage=0.9995,     # 对中文字符覆盖 99.5%
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    pad_piece="<pad>",
    split_by_unicode_script=True, # 按 Unicode script 切分
    split_by_whitespace=True,    # 中文无空格，关闭
    max_sentence_length=2048,
    remove_extra_whitespaces=True,
    normalization_rule_name="nmt_nfkc_cf"  # 统一规范化
)