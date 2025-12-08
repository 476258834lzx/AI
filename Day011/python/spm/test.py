import sentencepiece as spm

spm=spm.SentencePieceProcessor()
spm.Load('tokenizer.model')

print(spm.get_piece_size())      # 词表大小
print(spm.id_to_piece(100))      # ID→子词
print(spm.piece_to_id('壮士凌云')) # 子词→ID

# pieces = spm.Encode("毛泽东",out_type=str)
pieces = spm.encode_as_pieces("毛泽东")
print(pieces)
# ids = spm.Encode("毛泽东",out_type=int)
ids = spm.encode_as_ids("毛泽东")
print(ids)

