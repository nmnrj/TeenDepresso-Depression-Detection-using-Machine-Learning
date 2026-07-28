import pandas as pd

from preprocessing import preprocess_text

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

df = pd.read_csv("data/processed/final_dataset.csv")

print("Original Shape :", df.shape)

df.dropna(inplace=True)

print("Shape after removing null values :", df.shape)

print("\nCleaning Text...")

df["clean_text"] = df["text"].apply(preprocess_text)
df.dropna(subset=["clean_text"], inplace=True)

df = df[df["clean_text"].str.strip() != ""]
print("\nCleaning Completed!")

print("\nSample Results\n")

for i in range(5):

    print("=" * 70)
    print("Original :")
    print(df.loc[i, "text"])

    print("\nProcessed :")
    print(df.loc[i, "clean_text"])
    print()

df = df[["clean_text", "label"]]

df.to_csv(
    "data/processed/cleaned_dataset.csv",
    index=False
)

print("=" * 70)
print("Dataset Saved Successfully!")
print("=" * 70)

print(df.head())