from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I love coding",
    "I enjoy programming",
    "I hate bugs",
    "Hai Abhilash"
]

embeddings = model.encode(sentences)

for i, emb in enumerate(embeddings):
    print(f"Sentence: {sentences[i]}")
    print(f"Vector (first 5 values): {emb[:2]}")
    print()
print(embeddings)