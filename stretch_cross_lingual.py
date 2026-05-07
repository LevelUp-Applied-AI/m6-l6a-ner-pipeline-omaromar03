"""
Stretch 6B-S2: Cross-Lingual Embedding Comparison
==================================================
Extracts multilingual BERT embeddings for English and Arabic climate texts
and measures whether same-topic texts cluster together across languages.

Model: bert-base-multilingual-cased
Pooling: mean pooling over last hidden states (same as Lab 6B pipeline)
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# 1. Load and inspect data
# ---------------------------------------------------------------
DATA_PATH = "data/climate_articles.csv"
df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns.tolist())
print("Languages available:", df["language"].value_counts().to_dict())
print(df.head(2))

# NOTE: adjust the text column name if yours is different (e.g. 'content', 'article')
TEXT_COL = "text"   # <-- change if needed
LANG_COL = "language"

# ---------------------------------------------------------------
# 2. Select 10 English + 10 Arabic texts
# ---------------------------------------------------------------
# For best results, try to pick pairs that cover the same topics
# (e.g., both have an article about IPCC, both about renewable energy, etc.)
# The simplest approach: take the first 10 from each language.
# If your CSV has a 'topic' or 'category' column, use it to align pairs.

N = 10
en_df = df[df[LANG_COL] == "en"].head(N).reset_index(drop=True)
ar_df = df[df[LANG_COL] == "ar"].head(N).reset_index(drop=True)

assert len(en_df) == N, f"Need {N} English texts, found {len(en_df)}"
assert len(ar_df) == N, f"Need {N} Arabic texts, found {len(ar_df)}"

# Combine into one ordered dataframe: first 10 English, then 10 Arabic
all_df = pd.concat([en_df, ar_df]).reset_index(drop=True)
texts = all_df[TEXT_COL].tolist()
langs = all_df[LANG_COL].tolist()

print(f"\nLoaded {len(texts)} texts ({N} en + {N} ar)")

# ---------------------------------------------------------------
# 3. Load multilingual BERT
# ---------------------------------------------------------------
MODEL_NAME = "bert-base-multilingual-cased"
print(f"\nLoading {MODEL_NAME} (~680MB, first run will download)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ---------------------------------------------------------------
# 4. Mean-pooling embedding function
# ---------------------------------------------------------------
def mean_pool(last_hidden_state, attention_mask):
    """Mean pooling that ignores padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed(text: str) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled[0].numpy()

# ---------------------------------------------------------------
# 5. Embed all 20 texts
# ---------------------------------------------------------------
print("\nExtracting embeddings...")
embeddings = np.vstack([embed(t) for t in texts])
print(f"Embedding matrix shape: {embeddings.shape}")  # (20, 768)

# ---------------------------------------------------------------
# 6. 20x20 cosine similarity matrix
# ---------------------------------------------------------------
sim = cosine_similarity(embeddings)

# Save the matrix for the analysis writeup
np.save("similarity_matrix.npy", sim)

# ---------------------------------------------------------------
# 7. Heatmap
# ---------------------------------------------------------------
labels = [f"[{lang}] {text[:40]}" for text, lang in zip(texts, langs)]

plt.figure(figsize=(14, 12))
sns.heatmap(
    sim,
    xticklabels=labels,
    yticklabels=labels,
    cmap="RdYlBu_r",
    vmin=0, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 6},
    cbar_kws={"label": "Cosine similarity"},
)
plt.title("Cross-Lingual Cosine Similarity — bert-base-multilingual-cased\n"
          "(rows 0–9 = English, rows 10–19 = Arabic)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("cross_lingual_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cross_lingual_heatmap.png")

# ---------------------------------------------------------------
# 8. Comparative statistics for the analysis
# ---------------------------------------------------------------
en_idx = list(range(0, N))
ar_idx = list(range(N, 2 * N))

# Cross-lingual block (English rows × Arabic cols)
cross_block = sim[np.ix_(en_idx, ar_idx)]

# Within-language blocks (exclude diagonal of 1.0)
en_block = sim[np.ix_(en_idx, en_idx)]
ar_block = sim[np.ix_(ar_idx, ar_idx)]
en_within = en_block[np.triu_indices(N, k=1)]
ar_within = ar_block[np.triu_indices(N, k=1)]

# Diagonal of cross-block = paired translations (only meaningful if you aligned them)
paired_diag = np.diag(cross_block)

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (paste these into stretch_analysis.md)")
print("=" * 60)
print(f"Mean cross-lingual similarity (all en×ar pairs):  {cross_block.mean():.3f}")
print(f"Mean paired similarity (diagonal en[i] vs ar[i]): {paired_diag.mean():.3f}")
print(f"Mean within-English similarity:                   {en_within.mean():.3f}")
print(f"Mean within-Arabic similarity:                    {ar_within.mean():.3f}")

# Find the strongest and weakest cross-lingual pairs
print("\nTop 3 strongest cross-lingual pairs:")
flat = [(i, j, cross_block[i, j]) for i in range(N) for j in range(N)]
flat.sort(key=lambda x: -x[2])
for i, j, s in flat[:3]:
    print(f"  sim={s:.3f}")
    print(f"    EN: {texts[i][:80]}")
    print(f"    AR: {texts[N + j][:80]}")

print("\nTop 3 weakest cross-lingual pairs:")
for i, j, s in flat[-3:]:
    print(f"  sim={s:.3f}")
    print(f"    EN: {texts[i][:80]}")
    print(f"    AR: {texts[N + j][:80]}")

print("\nDone. Use the numbers above in your stretch2_analysis.md.")