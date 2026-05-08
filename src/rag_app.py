from datasets import load_dataset

# Load dataset
dataset = load_dataset("amazon_polarity")

# Print general dataset structure
print(dataset)

# Print one training example
print("\nFirst training example:\n")

print(dataset["train"][0])