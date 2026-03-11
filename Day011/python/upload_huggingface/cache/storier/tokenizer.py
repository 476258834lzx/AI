from transformers import PreTrainedTokenizer
import sentencepiece as spm
import os
import json
import shutil

#重载的PreTrainedTokenizer类
class SentencePieceTokenizer(PreTrainedTokenizer):

    def __init__(self, model_file, vocab_file, **kwargs):
        self._name_or_path = kwargs.get("name_or_path", ".")

        self._model_file = model_file
        self._vocab_file = vocab_file

        self.sp = spm.SentencePieceProcessor(model_file=f"{self._name_or_path}/{self._model_file}")

        with open(f"{self._name_or_path}/{self._vocab_file}", 'r', encoding='utf-8') as f:
            self.vocab = {line.strip().split('\t')[0]: i for i, line in enumerate(f)}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self.special_tokens={
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        }

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

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: list[int],
            token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        添加特殊 token 到输入序列
        - 单句: [BOS] + token_ids_0 + [EOS]
        - 双句: [BOS] + token_ids_0 + [EOS] + token_ids_1 + [EOS]
        """
        if token_ids_1 is None:
            # 单句情况
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            # 双句情况（如问答、句子对）
            return (
                    [self.bos_token_id] +
                    token_ids_0 +
                    [self.eos_token_id] +
                    token_ids_1 +
                    [self.eos_token_id]
            )

    def _tokenize(self, text):
        """
        用于分词
        """
        return self.sp.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """
        对词进行编码
        """
        # return self.sp.Encode(token)手动定义的<s>会识别为3个字符
        return self.vocab.get(token, self.vocab['<unk>'])

    def _convert_id_to_token(self, index):
        """
        对词进行解码
        """
        # return self.sp.Decode(index)手动定义的特殊字符不匹配
        return self.id_to_token.get(index, '<unk>')

    def convert_tokens_to_string(self, tokens):
        return self.sp.decode(tokens)

    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_py_src = os.path.join(self._name_or_path, "tokenizer.py")
        model_file_src = os.path.join(self._name_or_path, self._model_file)
        vocab_file_src = os.path.join(self._name_or_path, self._vocab_file)

        tokenizer_py_dst = os.path.join(save_directory, "tokenizer.py")
        model_file_dst = os.path.join(save_directory, self._model_file)
        vocab_file_dst = os.path.join(save_directory, self._vocab_file)

        shutil.copy2(tokenizer_py_src, tokenizer_py_dst)
        shutil.copy2(model_file_src, model_file_dst)
        shutil.copy2(vocab_file_src, vocab_file_dst)

        #AutoTokenizer映射到tokenizer.py中的SentencePieceTokenizer类
        tokenizer_config = {
            "bos_token": self.special_tokens["bos_token"],
            "eos_token": self.special_tokens["eos_token"],
            "pad_token": self.special_tokens["pad_token"],
            "unk_token": self.special_tokens["unk_token"],
            "tokenizer_class": "SentencePieceTokenizer",
            "model_file": self._model_file,
            "vocab_file": self._vocab_file,
            "auto_map": {
                "AutoTokenizer": [None, "tokenizer.SentencePieceTokenizer"]
            }
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        return self._vocab_file, self._model_file


if __name__ == '__main__':
    # 使用自定义的 Tokenizer
    tokenizer = SentencePieceTokenizer(
        model_file='tokenizer.model', vocab_file='tokenizer.vocab')

    # # 测试编码
    # text = '<s>这是一个测试句子</s>;'
    # tokens = tokenizer.tokenize(text)
    # print(tokens)
    # # 测试解码
    # decoded_text = tokenizer.convert_tokens_to_string(tokens)
    # print("Decoded text:", decoded_text)

    tokenizer.save_pretrained("./cache/storier")#会默认调用save_vocabulary
