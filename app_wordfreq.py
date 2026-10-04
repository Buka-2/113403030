import nltk

nltk.download("punkt_tab")
nltk.download("universal_tagset")
nltk.download('averaged_perceptron_tagger_eng')

sentence = """
For more than a century, the people of Trieste, in northeastern Italy, have performed a very specific ritual when they go for a swim.

At the La Lanterna city beach, where generations of Triestini have enjoyed a dip in their lunch break, or whiled away weekends on the pebbly shore, visitors first line up at a window to pay their entry fee of 1.20 euros — about $1.40.

Then they enter through one of two doors: women on the left, men to the right. Corridors lead to the changing areas, and from there it’s straight to the beach. This is where newbies might get a surprise — because it’s not just the changing rooms that are single-sex.

La Lanterna, also known as El Pedocin, is Europe’s last remaining gender-segregated beach. Step out onto the pebbles and you’ll be surrounded by women chatting in bikinis, or men playing cards, and never the twain shall meet. If you arrived with someone of the opposite sex, they’ll be on the other side of a 10-foot wall. Want to speak to them? Sure you can — if you swim out beyond the breaker. Only in the open waters are the genders allowed to mix.

El Pedocin has been a beloved haunt of the Triestini since it opened in the late 19th or early 20th century, but it made headlines in June when two tourists from Milan were less than impressed by its charms.


According to local media reports, which then made their way around the world, the couple paid to enter but were horrified when a fellow guest informed them of the separate gender rule.

The female tourist told the local woman that it was “absurd and medieval,” according to local paper Il Piccolo, before demanding a refund.
"""

print(f"原文：{sentence}")

tokens = nltk.word_tokenize(sentence)

print(f"分詞結果：{tokens}")

tagged = nltk.pos_tag(tokens, tagset="universal")
print(f"標註結果：{tagged}")


# === Stemming & Lemmatization（英文）===
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("wordnet")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("\n=== Stemming & Lemmatization ===")
wn_pos_map = {"VERB": "v", "NOUN": "n", "ADJ": "a", "ADV": "r"}
lemmatized = []
for word, pos in tagged:
    stem = stemmer.stem(word)
    wn_pos = wn_pos_map.get(pos, "n")
    lemma = lemmatizer.lemmatize(word, pos=wn_pos)
    lemmatized.append((lemma, pos))
    if word != stem or word != lemma:
        print(f"  {word:15} → Stem: {stem:15} Lemma: {lemma}")

print(f"\nLemmatization 結果：{lemmatized}")


# === Stopping & Filtering（使用 Lemma 結果）===
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
filtered = [(word, pos) for word, pos in lemmatized if word.lower() not in stop_words and word.isalpha()]
print(f"\n=== Stopping & Filtering ===")
print(f"過濾停用詞與標點後：{filtered}")


# =============================================================
# === ① 詞頻統計 + 文字雲 ===
# =============================================================
from collections import Counter

# 詞頻統計
words_only = [word for word, pos in filtered]
word_freq = Counter(words_only)

print("\n=== 詞頻統計 ===")
for word, count in word_freq.most_common():
    print(f"  {word:15} → {count} 次")

# 依詞性分組統計
pos_freq = Counter(pos for word, pos in filtered)
print("\n=== 詞性分佈 ===")
for pos, count in pos_freq.most_common():
    print(f"  {pos:10} → {count} 次")

# 文字雲
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # 用詞頻產生文字雲
    text = " ".join(words_only)
    text += "113403030 "
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.tight_layout()
    plt.savefig("wordcloud.png", dpi=150)
    plt.show()
    print("\n文字雲已儲存為 wordcloud.png")

except ImportError:
    print("\n⚠️ 請先安裝 wordcloud 和 matplotlib：")
    print("  pip install wordcloud matplotlib")