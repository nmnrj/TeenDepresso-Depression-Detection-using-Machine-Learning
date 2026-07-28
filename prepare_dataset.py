import pandas as pd

print("=" * 50)
print("Preparing Dataset")
print("=" * 50)

# Load the sampled dataset
df = pd.read_csv("data/processed/depression_50k.csv")

print("\nOriginal Shape :", df.shape)

# ----------------------------------------------------
# Drop unnecessary columns
# ----------------------------------------------------
columns_to_drop = [
    "Unnamed: 0",
    "subreddit",
    "upvotes",
    "created_utc",
    "num_comments"
]

df.drop(columns=columns_to_drop, inplace=True)

print("\nShape after dropping columns :", df.shape)

# ----------------------------------------------------
# Replace missing values
# ----------------------------------------------------
df["title"] = df["title"].fillna("")
df["body"] = df["body"].fillna("")

# ----------------------------------------------------
# Merge title and body
# ----------------------------------------------------
df["text"] = df["title"] + " " + df["body"]

# Remove unwanted spaces
df["text"] = df["text"].str.strip()

# ----------------------------------------------------
# Keep only text and label
# ----------------------------------------------------
df = df[["text", "label"]]

print("\nFinal Shape :", df.shape)

print("\nFirst 5 Rows\n")
print(df.head())

# ----------------------------------------------------
# Save dataset
# ----------------------------------------------------
df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("\nDataset saved successfully!")