from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
from products import products

model = SentenceTransformer("all-MiniLM-L6-v2")

# client = QdrantClient(":memory:")
client = QdrantClient(path="qdrant_data")

if not client.collection_exists("products"):
    client.create_collection(
        collection_name="products",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

points = []

for product in products:
    text = f"{product['name']} {product['description']} {product['category']}"

    vector = model.encode(text).tolist()

    points.append(PointStruct(
        id=product["id"], 
        vector=vector, 
        payload=product
    )
)

client.upsert(
    collection_name="products",
    points=points
)

print("✅ Products stored in Qdrant!")

client.close()
