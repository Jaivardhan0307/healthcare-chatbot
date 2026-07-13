import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# ── Step 1: Read the parquet file ─────────────────────────────────────────────
print("Reading parquet file...")
real_df = pd.read_parquet("data/real_faq_raw.parquet")

# ── Step 2: Keep only columns we need ─────────────────────────────────────────
real_df = real_df[["input", "output"]]

# ── Step 3: Rename to match our project format ────────────────────────────────
real_df = real_df.rename(columns={
    "input": "question",
    "output": "answer"
})

# ── Step 4: Drop empty rows ────────────────────────────────────────────────────
real_df = real_df.dropna(subset=["question", "answer"])

# ── Step 5: Keep only 500 rows ────────────────────────────────────────────────
real_df = real_df.head(500)

# ── Step 6: Save as faq.csv ───────────────────────────────────────────────────
real_df.to_csv("data/faq.csv", index=False)

print(f"✅ Done! faq.csv created with {len(real_df)} rows")
print("\nSample:")
print(real_df.head(3))