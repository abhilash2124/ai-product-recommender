from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
# client = QdrantClient(":memory:")
client = QdrantClient(path="qdrant_data")

# User query
query = input("Enter your requirement: ")

# Convert to embedding
query_vector = model.encode(query).tolist()

# Search in Qdrant
# results = client.query_points(
#     collection_name="products", 
#     query=query_vector, 
#     limit=3
#     )
# Detect category from query
category = None
if "phone" in query.lower():
    category = "phone"
elif "laptop" in query.lower():
    category = "laptop"

# Apply filter if detected
if category:
    results = client.query_points(
        collection_name="products",
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        ),
        limit=3
    )
else:
    results = client.query_points(
        collection_name="products",
        query=query_vector,
        limit=3
    )

print("\n🔍 Recommended Products:\n")

for res in results.points:
    product = res.payload
    print(f"📱 {product['name']}")
    print(f"   💬 {product['description']}")
    print(f"   💰 ₹{product['price']}")
    print(f"   ⭐ Score: {res.score}\n")

client.close()