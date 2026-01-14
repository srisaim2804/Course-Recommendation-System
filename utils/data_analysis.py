import os
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔢 Global variable
NUM_SENTENCES = 1000
EMBEDDING_FILE = "embeddings.npy"

# 1. Load DataFrame
with open("data.pkl", "rb") as f:
    df = pickle.load(f)

print("\n🧾 One sample row from the dataset:")
print(df.iloc[0])
print(df.iloc[1000])
print(df.iloc[2000])

# 2. Stats before filtering
total_rows = len(df)
duplicate_count = df.duplicated(subset=["reviews"]).sum()

print(f"\n🧾 Total rows in dataset: {total_rows}")
print(f"🔁 Number of duplicate reviews: {duplicate_count}")

# Institutions List
unique_institutions = df["institution"].nunique()
print(f"\n🏫 Number of unique institutions: {unique_institutions}")

# Optional: List the institution names
print("\n📍 Institutions:")
for inst in sorted(df["institution"].dropna().unique()):
    print(f"- {inst}")

# 📚 Total number of unique courses offered by each institution
print("\n📚 Total number of unique courses offered by each institution:")
course_counts = df.groupby("institution")["name"].nunique().sort_values(ascending=False)
for inst, count in course_counts.items():
    print(f"- {inst}: {count} course(s)")

# 🧮 Total number of reviews per course from each institution (formatted nicely)
print("\n📊 Detailed Summary: Reviews per Course by Institution\n")
reviews_per_course = df.groupby(["institution", "name"]).size().reset_index(name="review_count")
course_counts = reviews_per_course.groupby("institution")["name"].nunique()

for institution in sorted(reviews_per_course["institution"].unique()):
    total_courses = course_counts[institution]
    print(f"\n🏫 Institution: {institution} — {total_courses} course(s)")
    print("-" * (len(institution) + 20))
    inst_df = reviews_per_course[reviews_per_course["institution"] == institution]
    for _, row in inst_df.iterrows():
        print(f"   📘 {row['name']}: {row['review_count']} review(s)")

# 3. Show example of two duplicate reviews
duplicates_df = df[df.duplicated(subset=["reviews"], keep=False)]
if len(duplicates_df) >= 2:
    grouped = duplicates_df.groupby("reviews")
    for review_text, group in grouped:
        if len(group) >= 2:
            print("\n📋 Two duplicate reviews found:\n")
            print(f"🔸 Review 1:\n{group.iloc[0]['reviews']}")
            print(f"\n🔸 Review 2:\n{group.iloc[1]['reviews']}")
            break
    else:
        print("❌ No exact duplicate text found.")
else:
    print("❌ Not enough duplicate entries to show examples.")

# 4. Filter to unique reviews only
df_unique = df.drop_duplicates(subset=["reviews"]).reset_index(drop=True)
print(f"\n✨ Unique reviews retained: {len(df_unique)}")

# 5. Combine columns into a single string per row
def combine_fields(row):
    parts = [
        f"Review: {row['reviews']}",
        f"Course: {row['name']}",
        f"Institution: {row['institution']}",
        f"Rating: {row['rating']}",
        f"Reviewer: {row['reviewers']}",
        f"Date: {row['date_reviews']}",
    ]
    return " | ".join([str(p) for p in parts if pd.notnull(p)])

# 6. Apply to first NUM_SENTENCES of unique reviews
df_filtered = df_unique.iloc[:NUM_SENTENCES]
combined_texts = df_filtered.apply(combine_fields, axis=1).tolist()

print("\n📌 Example combined sentence from data:")
print(combined_texts[0])

# 7. Load model
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 8. Load or compute embeddings
if os.path.exists(EMBEDDING_FILE):
    print(f"\n📂 Found existing embeddings file: {EMBEDDING_FILE}, so loading from disk.")
    with open(EMBEDDING_FILE, "rb") as f:
        embeddings = np.load(f)
else:
    print(f"\n⚙️ Embeddings file not found. Encoding the first {NUM_SENTENCES} unique combined texts...")
    embeddings = model.encode(
        combined_texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    with open(EMBEDDING_FILE, "wb") as f:
        np.save(f, embeddings)
    print("✅ Embeddings saved to disk.")

# 9. Create FAISS index
dim = embeddings[0].shape[0]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

# 🔍 Search function
def search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    scores, indices = index.search(query_embedding, top_k)

    print(f"\n🔍 Query: {query}\n")
    for i, idx in enumerate(indices[0]):
        row = df_filtered.iloc[idx]
        print(f"Rank {i+1} (Score: {scores[0][i]:.4f})")
        print(f"Review: {row['reviews']}")
        print(f"Course: {row['name']} | Institution: {row['institution']}")
        print(f"Rating: {row['rating']} | By: {row['reviewers']} on {row['date_reviews']}\n")

# 🔁 Interactive search loop
# while True:
#     try:
#         user_query = input("\n🔎 Enter your question (or type 'exit' to quit): ")
#         if user_query.lower() in ["exit", "quit"]:
#             print("👋 Exiting search loop.")
#             break
#         search(user_query)
#     except KeyboardInterrupt:
#         print("\n👋 Exiting on keyboard interrupt.")
#         break