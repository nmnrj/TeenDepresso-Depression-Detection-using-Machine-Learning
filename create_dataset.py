import pandas as pd

# ================================
# Load Dataset
# ================================

print("Loading dataset...")

df = pd.read_csv("data/raw/depression_dataset.csv")

print("Dataset loaded successfully!")

print(f"\nOriginal Shape : {df.shape}")

# ================================
# Take Random 50,000 Samples
# ================================

df = df.sample(n=50000, random_state=42)

print(f"Sampled Shape : {df.shape}")

# ================================
# Save Dataset
# ================================

df.to_csv(
    "data/processed/depression_50k.csv",
    index=False
)

print("\n50K dataset saved successfully!")