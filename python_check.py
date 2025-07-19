import csv
import random

def generate_mock_data(rows, include_labels=True):
    data = []
    for _ in range(rows):
        row = []
        if include_labels:
            row.append(random.randint(0, 9))  # Random label
        row.extend(random.randint(0, 255) for _ in range(784))  # Pixel values
        data.append(row)
    return data

def write_csv(filename, data, include_header=True):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        if include_header:
            header = ["label"] + [f"pixel{i}" for i in range(784)] if len(data[0]) == 785 else [f"pixel{i}" for i in range(784)]
            writer.writerow(header)
        writer.writerows(data)

# === Paths ===
train_path = "./mnist_mock_train.csv"
test_path = "./mnist_mock_test.csv"

# === Generate and Save ===
train_data = generate_mock_data(10, include_labels=True)
test_data = generate_mock_data(10, include_labels=False)

write_csv(train_path, train_data)
write_csv(test_path, test_data)

print(f"Mock training and test files created:\n- {train_path}\n- {test_path}")
