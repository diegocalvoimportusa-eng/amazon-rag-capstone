from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------------
# Load Dataset
# -----------------------------------

dataset = load_dataset("amazon_polarity")

# Use smaller sample
sample_data = dataset["train"].select(range(1000))

# Convert to DataFrame
df = pd.DataFrame(sample_data)

# Combine title + content
df["full_review"] = df["title"].fillna("") + " " + df["content"].fillna("")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# -----------------------------------
# Load Embedding Model
# -----------------------------------

print("\nLoading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Generate Embeddings
# -----------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    df["full_review"].tolist(),
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)

# -----------------------------------
# Create FAISS Index
# -----------------------------------

dimension = embeddings.shape[1]

print("\nCreating FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index ready.")
print("Vectors stored:", index.ntotal)

# -----------------------------------
# Search Function
# -----------------------------------

def search_reviews(query, k=3):

    print("\n" + "=" * 60)
    print("USER QUERY")
    print("=" * 60)

    print(query)

    # Convert query to embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, k)

    print("\n" + "=" * 60)
    print("TOP MATCHING REVIEWS")
    print("=" * 60)

    retrieved_reviews = []

    for i, idx in enumerate(indices[0]):

        title = df.iloc[idx]["title"]
        review = df.iloc[idx]["content"]
        label = df.iloc[idx]["label"]
        distance = distances[0][i]

        sentiment = "Positive" if label == 1 else "Negative"

        retrieved_reviews.append({
            "title": title,
            "review": review,
            "sentiment": sentiment,
            "distance": distance
        })

        print(f"\nResult {i+1}")
        print("-" * 50)

        print("Title:", title)
        print("Sentiment:", sentiment)
        print("Distance:", distance)

        print("\nReview:")
        print(review)

    # -----------------------------------
    # Generate Final Response
    # -----------------------------------

    print("\n" + "=" * 60)
    print("GENERATED ASSISTANT RESPONSE")
    print("=" * 60)

    positive_reviews = [
        r for r in retrieved_reviews
        if r["sentiment"] == "Positive"
    ]

    negative_reviews = [
        r for r in retrieved_reviews
        if r["sentiment"] == "Negative"
    ]

    if positive_reviews:

        print("\nThis product category appears to have positive customer feedback.")

        print("\nPositive review highlights:")

        for r in positive_reviews:
            print(f"- {r['title']}")

    if negative_reviews:

        print("\nSome negative feedback was also found:")

        for r in negative_reviews:
            print(f"- {r['title']}")

    print("\nFinal Recommendation:")

    if len(positive_reviews) >= len(negative_reviews):

        print("This product category may be recommended based on retrieved reviews.")

    else:

        print("Use caution before recommending this product category.")

# -----------------------------------
# Run Example Query
# -----------------------------------

# -----------------------------------
# User Input
# -----------------------------------

user_query = input("\nEnter your product-related question: ")

search_reviews(user_query, k=3)