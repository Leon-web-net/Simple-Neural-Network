import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
test_path = "./MNIST_handwritten_dataset/test.csv"  # adjust if needed
output_dir = "./MNIST_images"
os.makedirs(output_dir, exist_ok=True)

# Load CSV (assuming first col = label, rest = pixels)
df = pd.read_csv(test_path,)
print("Dataset shape:", df.shape)

# Take 20 samples only
samples = df.sample(20, random_state=42).reset_index(drop=True)

for i in range(len(samples)):

    pixels = samples.iloc[i, :].values.reshape(28, 28)

    plt.imshow(pixels, cmap="gray")
    plt.axis("off")

    # Save as PNG
    save_path = os.path.join(output_dir, f"MNIST_digit_{i}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

print(f"✅ Saved {len(samples)} PNG images to {output_dir}")
