import os
import json
import shutil
import sentencepiece as spm
from transformers import PreTrainedTokenizer

class SentencePieceTokenizer(PreTrainedTokenizer):
    _auto_map = {"AutoTokenizer": [None, "tokenizer.SentencePieceTokenizer"]}

    def __init__(self, model_file, vocab_file, **kwargs):
        # 记录原始路径（用于加载模型和词汇表）
        self._name_or_path = kwargs.get("name_or_path", ".")
        self._model_file = model_file
        self._vocab_file = vocab_file

        # 加载 SentencePiece 模型和词汇表
        self.sp = spm.SentencePieceProcessor(model_file=f"{self._name_or_path}/{self._model_file}")
        with open(f"{self._name_or_path}/{self._vocab_file}", 'r', encoding='utf-8') as f:
            self.vocab = {line.strip().split('\t')[0]: i for i, line in enumerate(f)}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # 定义特殊 token
        special_tokens = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        }
        # 将特殊 token 和自定义字段传入父类，确保它们被记录到配置中
        kwargs.update(special_tokens)
        kwargs["model_file"] = model_file
        kwargs["vocab_file"] = vocab_file

        super().__init__(**kwargs)

        self.chat_template = """
            {%- for message in messages %}
                {%- if (message.role == "system") %}{{- '<s>system:\n'+ message.content + '</s>\n' }}{%- endif %}
                {%- if (message.role == "user") %}{{- '<s>user:\n'+ message.content + '</s>\n' }}{%- endif %}
            {%- endfor %}
            {{- '<s>assistant:\n' }}
        """

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        """分词，返回 token 字符串列表"""
        return self.sp.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """token 字符串 -> ID"""
        return self.vocab.get(token, self.vocab['<unk>'])

    def _convert_id_to_token(self, index):
        """ID -> token 字符串"""
        return self.id_to_token.get(index, '<unk>')

    def convert_tokens_to_string(self, tokens):
        """将 token 字符串列表合并为原始文本"""
        # 先将 token 字符串转为 ID，再调用 SentencePiece 解码
        token_ids = [self._convert_token_to_id(t) for t in tokens]
        return self.sp.decode(token_ids)

    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """保存词汇表和模型文件，并返回它们的路径"""
        os.makedirs(save_directory, exist_ok=True)

        # 复制模型文件
        model_src = os.path.join(self._name_or_path, self._model_file)
        model_dst = os.path.join(save_directory, self._model_file)
        shutil.copy2(model_src, model_dst)

        # 复制词汇表文件
        vocab_src = os.path.join(self._name_or_path, self._vocab_file)
        vocab_dst = os.path.join(save_directory, self._vocab_file)
        shutil.copy2(vocab_src, vocab_dst)

        # 复制 tokenizer.py（以便保存目录自包含）
        tokenizer_py_src = os.path.join(self._name_or_path, "tokenizer.py")
        tokenizer_py_dst = os.path.join(save_directory, "tokenizer.py")
        if os.path.exists(tokenizer_py_src):
            shutil.copy2(tokenizer_py_src, tokenizer_py_dst)

        # 返回文件路径元组（父类需要）
        return (vocab_dst, model_dst)


if __name__ == '__main__':
    # 使用自定义 Tokenizer
    tokenizer = SentencePieceTokenizer(
        model_file='tokenizer.model',
        vocab_file='tokenizer.vocab'
    )
    tokenizer.save_pretrained("./cache/storier")