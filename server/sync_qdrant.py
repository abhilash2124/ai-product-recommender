from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["ai_recommender"]
collection = db["products"]

# Qdrant
qdrant = QdrantClient(path="qdrant_data")

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection if not exists
if not qdrant.collection_exists("products"):
    qdrant.create_collection(
        collection_name="products",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Fetch products
products = list(collection.find())

points = []

for i, product in enumerate(products):
    text = f"{product['name']} {product['description']} {product['category']}"
    vector = model.encode(text).tolist()

    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "name": product["name"],
                "description": product["description"],
                "category": product["category"],
                "price": product["price"]
            }
        )
    )

# Upload to Qdrant
qdrant.upsert(
    collection_name="products",
    points=points
)

print("✅ MongoDB → Qdrant sync completed!")