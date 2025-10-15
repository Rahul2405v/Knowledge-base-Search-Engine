from app.embeddings import embedding_provider
import numpy as np

print("=== Debugging Embeddings ===")
print(f"OpenAI client available: {embedding_provider.openai_client is not None}")
print(f"Sentence model available: {embedding_provider.sentence_model is not None}")

# Test with different texts
test_texts = [
    "hello world", 
    "python programming",
    "machine learning",
    "database query"
]

print(f"\nTesting with {len(test_texts)} different texts...")
embeddings = embedding_provider.get_embeddings(test_texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Check if embeddings are actually different
print("\nFirst 5 values of each embedding:")
for i, emb in enumerate(embeddings):
    print(f"Text {i+1} ('{test_texts[i]}'): {emb[:5]}")

# Calculate similarities between embeddings
print("\nSimilarity matrix:")
for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        if i <= j:
            emb1 = np.array(embeddings[i])
            emb2 = np.array(embeddings[j])
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"Text {i+1} vs Text {j+1}: {similarity:.4f}")

print("\n=== End Debug ===")