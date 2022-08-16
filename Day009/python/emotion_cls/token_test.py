from transformers import BertTokenizer

#加载字典和分词工具
token=BertTokenizer.from_pretrained("bert-base-chinese")
# print(token)

# sents=['书中有7页的缺页，确切的说是7页的白纸，你们倒是省墨了，可苦了我，看起来没有连贯性，有时不知所云，故事情节连不上，让人非常的讨厌。',
#        '酒店设施陈旧，浴缸排水不畅，入住无房，一间16：00，一间22：00，早餐差',
#        '房间干净整洁，位置也好，就是缺少部电梯，有点遗憾']
#
# #批量语句编码
# out=token.batch_encode_plus(
#     batch_text_or_text_pairs=[sents[0],sents[1],sents[2]],
#     add_special_tokens=True,
#     #当语句长度大于nax_length时，进行截断
#     truncation=True,
#     #一律补0到max_length长度
#     padding="max_length",
#     max_length=20,
#     #返回类型tf,pt,np,默认返回list
#     return_tensors=None,
#     #返回attention_mask
#     return_attention_mask=True,
#     #返回token_type_ids
#     return_token_type_ids=True,
#     #返回special_tokens_mask特殊符号标识
#     return_special_tokens_mask=True,
#     #返回每个词的起始位置，只能在BertTokenizerFast模型中使用
#     # return_offsets_mapping=True,
#     #返回length标识的长度
#     return_length=True
# )
#
# # print(out)
# # input_ids句子的位置编码
# # token_type_ids第一个句子和特殊符号的位置是0，第二个句子为1，传入元组型句子段时，上半段为0，下半段为1
# # special_tokens_mask特殊符号的位置是1，其余为0
# # attention_mask pad的位置是0，其余为1
# for k,v in out.items():
#     print(k,":",v)
#
# print(token.decode(out["input_ids"][0]))

#获取字典
vocab=token.get_vocab()
print(type(vocab),len(vocab),"饕" in vocab)
#添加新词
token.add_tokens(new_tokens=["阳光","雨露"])
#添加新的符号
token.add_special_tokens({"eos_token":"[EOS]"})
vocab=token.get_vocab()
#{"阳光":位置编码，...}
print(type(vocab),len(vocab),"阳光" in vocab,"[EOS]" in vocab)
print(vocab["阳光"],vocab["雨露"],vocab["[EOS]"])

#编码新句子
out=token.encode(
    text="阳光和雨露[EOS]",
    text_pair=None,
    truncation=True,
    padding="max_length",
    max_length=8,
    add_special_tokens=True,
    return_tensors=None
)
print(out)
print(token.decode(out))