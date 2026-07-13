import pandas as pd

# Read the file
df = pd.read_csv("data/real_symptoms_raw.csv")

# Combine all 17 symptom columns into one column
symptom_cols = ['Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5',
                'Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10',
                'Symptom_11','Symptom_12','Symptom_13','Symptom_14','Symptom_15',
                'Symptom_16','Symptom_17']

df["symptoms"] = df[symptom_cols].apply(
    lambda row: ", ".join([v.strip() for v in row if pd.notna(v)]),
    axis=1
)

# Keep only disease and symptoms
df = df[["Disease", "symptoms"]]
df = df.rename(columns={"Disease": "disease"})

# Remove duplicate diseases
df = df.drop_duplicates(subset=["disease"])

# Save
df.to_csv("data/symptom_disease.csv", index=False)

print(f"✅ Done! {len(df)} diseases saved")
print(df.head(5))