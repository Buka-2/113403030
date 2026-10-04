import nltk

nltk.download("punkt_tab")
nltk.download("universal_tagset")

sentence = """
At eight o'clock on Thursday morning.
Arthur didn't feel very good.
"""

print(f"原文：{sentence}")

tokens = nltk.word_tokenize(sentence)


print(f"分詞結果：{tokens}")

tagged = nltk.pos_tag(tokens, tagset="universal")
print(f"標註結果：{tagged}")


import jieba
import jieba.posseg as pseg

sentence2 = """
禮拜四早上八點鐘。
阿瑟覺得不太舒服。
"""

print(f"原文：{sentence2}")

tokens2 = jieba.lcut(sentence2)
print(f"分詞結果：{tokens2}")

# 自訂映射到簡化標籤
tag_map = {
    "n": "NOUN",
    "nr": "NOUN",
    "ns": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "d": "ADV",
    "r": "PRON",
    "p": "ADP",
    "c": "CONJ",
    "m": "NUM",
    "t": "NOUN",
    "u": "PRT",
    "x": ".",
}

tagged2 = pseg.lcut(sentence2)
tagged2 = [(word, tag_map.get(flag, "X")) for word, flag in tagged2]
print(f"標註結果：{tagged2}")